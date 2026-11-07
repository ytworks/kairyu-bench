# Exact dependent scalar archive

Original source is mandatory; AV and faststore are not dependencies. Preserve all original records. Fresh existing editor.verify checked all 12 files, candidate bytes, and the candidate stream with the declared two-byte edit restored against original source SHA. The historical manifest lacked hashes; that fact remains true and the new full verification is recorded separately. No physical 62GB restore was made.

To reconstruct, first sha256sum -c SHA256SUMS in this archive and BASE-SHA256SUMS in the original base. Decompress spec.json.gz to a fresh metadata path. Use the archived source with the existing editor create <original-base> <new-checkpoint> --spec <spec> --manifest <new-manifest> --apply. Run editor verify on that new manifest, then compare all 12 reconstructed files against CANDIDATE-SHA256SUMS.
