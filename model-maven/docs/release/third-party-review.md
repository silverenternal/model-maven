# Third-party and licensing release checklist

This is an open decision checklist for maintainers. It is not legal advice, a
license grant, or a claim that the listed review is complete.

- [ ] Select and approve a root project license; add the corresponding license
  text and Cargo/Python package metadata only after that decision.
- [ ] Inventory direct Rust dependencies from `Cargo.lock`, their transitive
  dependencies, and license identifiers from authoritative package metadata.
- [ ] Inventory Python SDK build dependencies and any packages used by examples
  or release tooling; distinguish build-time from runtime dependencies.
- [ ] Review bundled worker source, examples, fixtures, fonts/assets, and copied
  snippets for ownership and redistribution terms.
- [ ] Decide whether to ship a generated third-party notices/SBOM document and
  define its reproducible generation and review process.
- [ ] Confirm trademark, project-name, export-control, privacy, and security
  disclosure decisions appropriate to the intended distribution channels.
- [ ] Have qualified maintainers/legal reviewers approve the final materials
  before any public binary or package publication.

The current release workflow only creates a temporary CI artifact retained for
30 days. It does not publish a GitHub Release, crates.io package, Python index
release, or other public distribution.
