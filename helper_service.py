"""
Helper service for TriSAFE's 2-of-3 threshold scheme.

Each HelperService holds exactly one threshold share (the coordinating
aggregator holds none) and is responsible for:

  1. creating and encrypting its discrete-Gaussian DP noise share, and
  2. releasing a transcript-bound partial decryption ONLY for the round's
     final noised ciphertext.
"""

import math
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from phe_mechanism import DiscreteGaussian, EncryptedPackedValue


class HelperService:
    """Helper node holding one threshold share (aggregator holds none)."""

    def __init__(self, helper_id: str, share_info: Dict[str, Any], phe_config,
                 public_key: Optional[Any] = None,
                 threshold_paillier: Optional[Any] = None):
        """
        Args:
            helper_id:   e.g. "H1", "H2", "H3".
            share_info:  dict with 'share_value' and (optionally) 'share_id',
                         as produced by ThresholdPaillier._generate_threshold_shares.
            phe_config:  a ThresholdPaillierConfig (provides weight_scale,
                         fixed_point_scale, slots_per_ciphertext, packing_base).
            public_key:  optional Paillier public key used to encrypt noise.
            threshold_paillier: optional ThresholdPaillier instance; if given its
                         public key / packing are reused.
        """
        self.helper_id = helper_id
        self.share = share_info['share_value']
        self.share_id = share_info.get('share_id', helper_id)
        self.config = phe_config

        self.threshold_paillier = threshold_paillier
        if public_key is None and threshold_paillier is not None:
            public_key = getattr(threshold_paillier, 'public_key', None)
        self.public_key = public_key

        # A helper releases a decryption share only for transcripts it has been
        # authorized for (i.e. the final noised-ciphertext digest of a round).
        self.authorized_transcripts = set()

        # Bookkeeping of the most recent noise sample (auditability).
        self.last_noise: Optional[np.ndarray] = None

        self.logger = logging.getLogger(f'Helper.{helper_id}')

    # ------------------------------------------------------------------ #
    # Noise generation
    # ------------------------------------------------------------------ #
    def generate_noise_share(self, dimension: int, sigma_real: float,
                             round_id: int):
        """Generate and encrypt this helper's discrete-Gaussian noise share.

        Each helper samples with per-coordinate variance
        ``(S_alpha * S_fp * sigma_real / sqrt(2))**2`` so that two honest
        helpers together meet the calibrated target variance.
        """
        scale = (self.config.weight_scale * self.config.fixed_point_scale
                 * sigma_real / math.sqrt(2))
        variance = scale ** 2

        noise = DiscreteGaussian.sample(
            mean=0,
            variance=variance,
            size=dimension,
            seed=self._get_seed(round_id),
        )
        self.last_noise = noise
        self.logger.debug(f"sampled noise: dim={dimension}, std={scale:.2f}")

        return self._encrypt_vector(noise)

    # ------------------------------------------------------------------ #
    # Transcript-bound partial decryption
    # ------------------------------------------------------------------ #
    def authorize(self, transcript_hash: str) -> None:
        """Authorize this helper to release a share for a given round digest.

        In deployment the coordinator publishes the final noised-ciphertext
        digest; the helper checks that its identity and noise-ciphertext hash
        are included before authorizing. Here we record the digest directly.
        """
        if transcript_hash:
            self.authorized_transcripts.add(transcript_hash)

    def partial_decrypt(self, ciphertext, transcript_hash: str) -> Tuple[Dict, Dict]:
        """Provide a transcript-bound partial decryption plus a validity proof.

        Refuses to act on transcript-inconsistent (e.g. un-noised or
        unauthorized) ciphertexts, mirroring the paper's helper behavior.
        """
        if not self._verify_transcript(transcript_hash):
            raise ValueError(
                f"{self.helper_id}: invalid/unauthorized transcript - refusing decryption"
            )

        partial = self._compute_partial_decryption(ciphertext)
        proof = self._generate_share_proof(partial, transcript_hash)
        self.logger.debug(f"released partial decryption for transcript {transcript_hash[:12]}...")
        return partial, proof

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_seed(self, round_id: int) -> int:
        """Deterministic per-round sampler seed (reproducible; not for deployment).

        Matches DistributedNoiseGenerator's seeding so helper noise is
        reproducible across the prototype. In deployment each helper derives
        its seed from an independent hardware root via HKDF.
        """
        seed_string = f"{self.helper_id}_{round_id}_noise"
        return int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16)

    def _encrypt_vector(self, noise: np.ndarray):
        """Encrypt the integer noise vector under the Paillier public key.

        Noise is encrypted coordinate-wise (one ciphertext per coordinate).
        This deliberately avoids the packed slot-bound check in
        ``ThresholdPaillier.encrypt_packed``: integer-domain DP noise can
        exceed a single packing slot, and packing it would corrupt adjacent
        slots. If no public key is bound, the raw integer noise is returned so
        a caller/test harness can encrypt it.
        """
        if self.public_key is None:
            return noise
        return [self.public_key.encrypt(int(v)) for v in noise.tolist()]

    def _verify_transcript(self, transcript_hash: str) -> bool:
        """Release a share only for a well-formed, explicitly authorized digest.

        A helper refuses any request that is not bound to a transcript it has
        been authorized for (e.g. an un-noised or unauthorized ciphertext).
        """
        if not transcript_hash:
            return False
        return transcript_hash in self.authorized_transcripts

    def _compute_partial_decryption(self, ciphertext) -> Dict[str, Any]:
        """Compute a (simulated) share-dependent partial decryption.

        Real Damgaard-Jurik partials are ``c^(2*Delta*s_i) mod N^2``. Here we
        produce a deterministic share-bound value with the same structure so
        the protocol flow is exercisable; reconstruction in this prototype is
        handled by ThresholdPaillier.threshold_decrypt.
        """
        # Extract an integer representation of the ciphertext.
        ct_int = None
        enc = getattr(ciphertext, 'encrypted_value', ciphertext)
        if hasattr(enc, 'ciphertext'):
            try:
                ct_int = enc.ciphertext(False)
            except TypeError:
                ct_int = enc.ciphertext()
        if ct_int is None:
            ct_int = int(hashlib.sha256(str(enc).encode()).hexdigest(), 16)

        modulus = (self.public_key.n ** 2) if self.public_key is not None else (2 ** 521 - 1)
        partial = pow(int(ct_int), int(self.share), int(modulus))

        return {
            'helper_id': self.helper_id,
            'share_id': self.share_id,
            'partial': partial,
        }

    def _generate_share_proof(self, partial: Dict[str, Any],
                              transcript_hash: str) -> Dict[str, Any]:
        """NIZK of share validity (placeholder), bound to the round transcript."""
        material = f"{self.helper_id}|{self.share_id}|{partial.get('partial')}|{transcript_hash}"
        return {
            'type': 'share_validity',
            'helper_id': self.helper_id,
            'transcript': transcript_hash,
            'commitment': hashlib.sha256(material.encode()).hexdigest(),
        }


if __name__ == "__main__":
    # Lightweight smoke test (no heavy key generation): exercises the noise
    # and transcript-binding paths with the raw-noise (no public key) branch.
    logging.basicConfig(level=logging.DEBUG)

    class _Cfg:
        weight_scale = 2 ** 16
        fixed_point_scale = 2 ** 20

    helper = HelperService("H1", {'share_value': 12345, 'share_id': 1}, _Cfg())

    noise = helper.generate_noise_share(dimension=8, sigma_real=0.32, round_id=0)
    print("noise share (first 4):", np.asarray(noise)[:4])

    digest = hashlib.sha256(b"round-0-noised-ciphertext").hexdigest()
    try:
        helper.partial_decrypt(ciphertext=98765, transcript_hash=digest)
        print("ERROR: decrypted without authorization")
    except ValueError as e:
        print("correctly refused unauthorized transcript:", e)

    helper.authorize(digest)
    partial, proof = helper.partial_decrypt(ciphertext=98765, transcript_hash=digest)
    print("partial share_id:", partial['share_id'], "proof type:", proof['type'])
