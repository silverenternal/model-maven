# Upgrade and rollback notes

This development snapshot has no automatic updater and does not replace an
existing binary. Install a candidate into a separate prefix, for example:

```sh
cargo install --path . --locked --root "$HOME/.local/model-maven-candidate"
"$HOME/.local/model-maven-candidate/bin/m-model" --version
```

For a review bundle, verify the published SHA-256 file before extracting it.
Keep the previous executable and back up `.model-maven/`, model source files,
environment lock files, and run/checkpoint artifacts before using commands
that modify them. To roll back, select the previous executable explicitly and
restore the project backup if a newer CLI has modified persisted state.

For a Linux review bundle, install each version into its own directory so a
candidate never overwrites the previous binary:

```sh
sha256sum -c model-maven-0.1.0-linux-x86_64.tar.gz.sha256
mkdir -p "$HOME/.local/opt/model-maven/0.1.0" "$HOME/.local/bin"
tar -xzf model-maven-0.1.0-linux-x86_64.tar.gz -C "$HOME/.local/opt/model-maven/0.1.0"
install -m 755 "$HOME/.local/opt/model-maven/0.1.0/model-maven-0.1.0-linux-x86_64/m-model" \
  "$HOME/.local/opt/model-maven/0.1.0/m-model"
ln -sfn "$HOME/.local/opt/model-maven/0.1.0/m-model" "$HOME/.local/bin/m-model"
"$HOME/.local/bin/m-model" --version
```

To roll back the executable, point `~/.local/bin/m-model` to the previous
versioned path. The bundle includes an optional SDK wheel; install it only in
the Python environment used for extension development, for example with
`python -m pip install ./model_maven_sdk-0.1.0-py3-none-any.whl`.

No automatic state migration, cross-version checkpoint restoration, or
backward-compatibility guarantee is promised for this development snapshot.
The tool rejects known unsupported state versions, but that is not a substitute
for a backup.
