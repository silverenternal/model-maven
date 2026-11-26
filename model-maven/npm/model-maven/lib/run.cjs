'use strict';

const { spawnSync } = require('node:child_process');
const { constants } = require('node:os');
const path = require('node:path');
const fs = require('node:fs');

module.exports = function run(name) {
  if (name === 'm-model-tui' && process.argv.length === 3
      && ['--workspace-plugin-dir', '--examples-dir'].includes(process.argv[2])) {
    try {
      const item = process.argv[2] === '--examples-dir' ? 'examples' : 'nvim';
      const directory = fs.realpathSync(path.join(__dirname, '..', item));
      if (!fs.statSync(directory).isDirectory()) throw new Error('not a directory');
      process.stdout.write(`${directory}\n`);
    } catch (error) {
      const description = process.argv[2] === '--examples-dir' ? 'examples' : 'Neovim plugin';
      console.error(`The Model-Maven ${description} directory is unavailable: ${error.message}`);
      process.exitCode = 1;
    }
    return;
  }
  const supported = new Set([
    'linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64', 'win32-x64', 'win32-arm64',
  ]);
  const platform = `${process.platform}-${process.arch}`;
  if (!supported.has(platform)) {
    console.error(`Model-Maven does not provide a binary for ${platform}.`);
    process.exitCode = 1;
    return;
  }
  let runtimeGlibc = null;
  if (process.platform === 'linux') {
    try {
      runtimeGlibc = process.report?.getReport()?.header?.glibcVersionRuntime || null;
    } catch (_) {
      runtimeGlibc = null;
    }
  }
  if (process.platform === 'linux' && !runtimeGlibc) {
    console.error('Could not detect glibc in the Node runtime report. Model-Maven Linux binaries require a compatible glibc system; musl/Alpine is not supported.');
    process.exitCode = 1;
    return;
  }
  const launcherPackage = require('../package.json');
  const packageName = `${launcherPackage.name}-${platform}`;
  let root;
  try {
    root = path.dirname(require.resolve(`${packageName}/package.json`));
  } catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND') throw error;
    console.error(`The ${packageName} binary package is missing. Reinstall with npm optional dependencies enabled.`);
    process.exitCode = 1;
    return;
  }
  const nativePackage = require(path.join(root, 'package.json'));
  const expectedWorkspace = launcherPackage.modelMaven?.workspace;
  const nativeWorkspace = nativePackage.modelMaven?.workspace;
  const validWorkspace = (value) => value && value.algorithm === 'sha256-file-map-v1'
    && typeof value.digest === 'string' && /^[0-9a-f]{64}$/.test(value.digest);
  if (nativePackage.name !== packageName || nativePackage.version !== launcherPackage.version
      || !validWorkspace(expectedWorkspace) || !validWorkspace(nativeWorkspace)
      || expectedWorkspace.digest !== nativeWorkspace.digest) {
    console.error(`The ${packageName} package does not match this Model-Maven workspace, or package identity is missing. Reinstall matching packages from the same release.`);
    process.exitCode = 1;
    return;
  }
  const expectedSources = launcherPackage.modelMaven?.sourceCheckouts;
  if (!expectedSources && launcherPackage.private !== true) {
    console.error('This Model-Maven package has no source checkout identity. Reinstall a complete release.');
    process.exitCode = 1;
    return;
  }
  if (expectedSources) {
    const nativeSources = nativePackage.modelMaven?.sourceCheckouts;
    const equalCheckout = (key) => expectedSources[key] && nativeSources?.[key]
      && typeof expectedSources[key].revision === 'string'
      && /^[0-9a-f]{40}$/.test(expectedSources[key].revision)
      && expectedSources[key].dirty === false && nativeSources[key].dirty === false
      && expectedSources[key].revision === nativeSources[key].revision;
    if (!equalCheckout('backend') || !equalCheckout('workspace')) {
      console.error(`The ${packageName} package was staged from different or unverified source checkouts. Reinstall packages from the same release.`);
      process.exitCode = 1;
      return;
    }
  }
  if (process.platform === 'linux') {
    const required = nativePackage.modelMaven?.glibcRequirements?.[name];
    const parseVersion = (value) => typeof value === 'string' && /^\d+\.\d+$/.test(value)
      ? value.split('.').map(Number) : null;
    const actualVersion = parseVersion(runtimeGlibc);
    const requiredVersion = parseVersion(required);
    if (!actualVersion || !requiredVersion) {
      console.error(`Cannot verify the glibc requirement for ${name}; the runtime or native package has missing or invalid version metadata. Reinstall a complete matching release or build for this system.`);
      process.exitCode = 1;
      return;
    }
    if (actualVersion[0] < requiredVersion[0]
        || (actualVersion[0] === requiredVersion[0] && actualVersion[1] < requiredVersion[1])) {
      console.error(`${name} requires glibc ${required} or newer; this system reports glibc ${runtimeGlibc}. Use a compatible system or build Model-Maven for this environment.`);
      process.exitCode = 1;
      return;
    }
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
