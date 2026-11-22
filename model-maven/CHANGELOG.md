# Changelog

All notable changes to Model-Maven will be documented here. Until the first
reviewed release, entries describe the current development snapshot and do not
imply a stable compatibility commitment.

## Unreleased

- Added a review-only Linux x86_64 release candidate workflow and a clean
  Cargo-package installation smoke check.
- The CLI embeds its Python worker. The optional `model-maven-sdk` is distributed
  as a separate wheel in the review bundle; it is not required to use the CLI.
- Current scope is an early native PyTorch development tool. See
  [`docs/release/compatibility.md`](docs/release/compatibility.md) for the
  tested platform boundary and known limitations.
- Project licensing and third-party notice decisions remain open. No license
  grant is made by this development snapshot.
