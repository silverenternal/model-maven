local M = {}

function M.check()
  vim.health.start('Model-Maven')
  if vim.fn.has('nvim-0.10') == 1 then
    vim.health.ok('Neovim provides the required asynchronous process API.')
  else
    vim.health.error('Neovim 0.10 or newer is required.')
    -- The workspace itself uses vim.uv and other 0.10 APIs. Stop here so
    -- health reports the compatibility boundary instead of failing midway.
    return
  end
  local mouse = vim.o.mouse or ''
  if not mouse:find('a', 1, true) and not mouse:find('n', 1, true) then
    vim.health.warn('Normal-mode mouse input is disabled in Neovim. Use Tab/arrow keys and Enter in the graph, M to reorder, c to connect, or enable mouse input in your own configuration.')
  else
    vim.health.info('Normal-mode mouse input is enabled in Neovim; terminal and multiplexer support is not verified.')
  end
  if vim.fn.exists(':ModelMaven') ~= 2 then
    vim.health.warn('The :ModelMaven command is unavailable. Check whether the plugin was loaded; require("model_maven").open() can open it directly.')
  elseif not vim.g.loaded_model_maven_workspace then
    vim.health.warn('A :ModelMaven command exists, but this plugin did not register it. Check for a command name collision before invoking it.')
  else
    vim.health.info('The plugin reports that it registered :ModelMaven; another plugin could still replace that command later.')
  end
  for _, name in ipairs({ 'select', 'input' }) do
    local implementation = vim.ui[name]
    local info = type(implementation) == 'function' and debug.getinfo(implementation, 'S') or nil
    local source = info and info.source
    if type(source) == 'string' then
      source = vim.fn.strcharpart(source:gsub('[%c]', ' '), 0, 512)
      vim.health.info('vim.ui.' .. name .. ' provider: ' .. source .. '. Its behavior is not verified by this check.')
    else
      vim.health.warn('vim.ui.' .. name .. ' is unavailable or its provider cannot be identified; interactive prompts may not work.')
    end
  end
  local ok, plugin = pcall(require, 'model_maven')
  if not ok or type(plugin) ~= 'table' or plugin.workspace_api_version ~= 1 or type(plugin.open) ~= 'function' then
    vim.health.error('The loaded Model-Maven module does not provide the expected workspace API; check plugin installation and runtime path.')
  else
    local info = debug.getinfo(plugin.open, 'S')
    local source = info and info.source or ''
    local loaded_path
    if source:sub(1, 1) == '@' then
      local resolved_ok, resolved = pcall(vim.uv.fs_realpath, source:sub(2))
      loaded_path = resolved_ok and resolved or vim.fs.normalize(source:sub(2))
    end
    if loaded_path then
      vim.health.info('Loaded workspace module: ' .. loaded_path)
    else
      vim.health.warn('The loaded workspace source path is unavailable; duplicate installations cannot be fully checked.')
    end
    local distinct, paths = {}, {}
    for _, candidate in ipairs(vim.api.nvim_get_runtime_file('lua/model_maven/init.lua', true)) do
      local resolved_ok, resolved = pcall(vim.uv.fs_realpath, candidate)
      resolved = resolved_ok and resolved or vim.fs.normalize(candidate)
      if not distinct[resolved] then
        distinct[resolved] = true
        paths[#paths + 1] = resolved
      end
    end
    if #paths > 1 then
      vim.health.warn('Multiple Model-Maven workspace installations are on runtimepath. The loaded module may differ from the one you expect.')
      for _, candidate in ipairs(paths) do vim.health.info('Workspace candidate: ' .. candidate) end
    elseif loaded_path and #paths == 1 and loaded_path ~= paths[1] then
      vim.health.warn('The loaded workspace module differs from the runtimepath candidate; check your plugin manager or Lua module cache.')
    end
  end
  local config
  if ok and type(plugin) == 'table' and type(plugin.get_config) == 'function' then
    local config_ok, value = pcall(plugin.get_config)
    if not config_ok then
      vim.health.error('Could not read Model-Maven setup() configuration: ' .. tostring(value))
      return
    end
    config = value
  end
  local executable = type(config) == 'table' and config.cli or vim.env.MODEL_MAVEN_CLI or 'm-model'
  if ok and type(plugin) == 'table' and type(plugin.resolve_cli) == 'function' then
    local resolved_ok, resolved = pcall(plugin.resolve_cli, executable)
    if not resolved_ok then
      vim.health.error('Could not resolve the configured CLI: ' .. tostring(resolved))
      return
    end
    executable = resolved
  end
  if type(executable) ~= 'string' or executable == '' then
    vim.health.error('Configured CLI must be a nonempty executable path or command.')
  elseif vim.fn.executable(executable) == 1 then
    vim.health.ok('CLI executable found: ' .. executable)
    vim.health.info('Checks setup() configuration resolved from the current editor directory; open() can override it. Project and runtime compatibility are checked separately; no model was imported.')
  else
    vim.health.error('Configured CLI was not found: ' .. executable .. '. Install it or set setup({ cli = ... }) / MODEL_MAVEN_CLI.')
  end
end

return M
