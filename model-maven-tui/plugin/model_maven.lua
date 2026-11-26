if vim.g.loaded_model_maven_workspace then return end
if vim.fn.exists(':ModelMaven') == 2 then
  vim.notify('Model-Maven did not define :ModelMaven because that user command already exists.', vim.log.levels.WARN)
  return
end
vim.api.nvim_create_user_command('ModelMaven', function(args)
  local ok, plugin = pcall(require, 'model_maven')
  if not ok or type(plugin) ~= 'table' or plugin.workspace_api_version ~= 1 or type(plugin.open) ~= 'function' then
    vim.notify('Could not load a compatible Model-Maven workspace. Check :checkhealth model_maven and the plugin runtimepath. '
      .. (not ok and tostring(plugin) or 'The loaded module does not expose workspace API 1.'), vim.log.levels.ERROR)
    return
  end
  local opened, error_message = pcall(plugin.open, { project = args.args ~= '' and args.args or nil })
  if not opened then
    vim.notify('Could not open Model-Maven. Check :messages and :checkhealth model_maven: ' .. tostring(error_message), vim.log.levels.ERROR)
  end
end, { nargs = '?', complete = 'dir', desc = 'Open the Model-Maven workspace' })
vim.g.loaded_model_maven_workspace = true
