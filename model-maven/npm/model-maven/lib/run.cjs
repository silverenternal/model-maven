'use strict';

const { spawnSync } = require('node:child_process');
const { constants } = require('node:os');
const path = require('node:path');

module.exports = function run(name) {
  const supported = new Set([
    'linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64', 'win32-x64', 'win32-arm64',
  ]);
  const platform = `${process.platform}-${process.arch}`;
  if (!supported.has(platform)) {
    console.error(`Model-Maven does not provide a binary for ${platform}.`);
    process.exitCode = 1;
    return;
  }
  if (process.platform === 'linux' && !process.report?.getReport().header.glibcVersionRuntime) {
    console.error('Model-Maven Linux binaries require glibc; musl/Alpine is not currently supported.');
    process.exitCode = 1;
    return;
  }
  const packageName = `${require('../package.json').name}-${platform}`;
  let root;
  try {
    root = path.dirname(require.resolve(`${packageName}/package.json`));
  } catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND') throw error;
    console.error(`The ${packageName} binary package is missing. Reinstall with npm optional dependencies enabled.`);
    process.exitCode = 1;
    return;
  }
  const suffix = process.platform === 'win32' ? '.exe' : '';
  const executable = path.join(root, 'bin', name + suffix);
  const env = { ...process.env };
  if (name === 'm-model-tui' && !env.MODEL_MAVEN_CLI) {
    env.MODEL_MAVEN_CLI = path.join(root, 'bin', 'm-model' + suffix);
  }
  const result = spawnSync(executable, process.argv.slice(2), { stdio: 'inherit', env });
  if (result.error) {
    console.error(`Could not start ${name}: ${result.error.message}`);
    process.exitCode = 1;
  } else if (result.signal) {
    process.exitCode = 128 + (constants.signals[result.signal] || 0);
  } else {
    process.exitCode = result.status === null ? 1 : result.status;
  }
};
