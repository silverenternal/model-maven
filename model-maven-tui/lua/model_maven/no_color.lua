-- Used only by the standalone launcher when NO_COLOR is present. Existing
-- Neovim sessions keep their own theme and global highlights untouched.
local M = {}

function M.apply()
  local ok, groups = pcall(vim.api.nvim_get_hl, 0, {})
  if not ok then return end
  for name, attributes in pairs(groups) do
    -- Preserve links and non-colour attributes such as bold and reverse.
    if type(attributes) == 'table' and not attributes.link then
      attributes.fg, attributes.bg, attributes.sp = nil, nil, nil
      attributes.ctermfg, attributes.ctermbg = nil, nil
      attributes.fg_indexed, attributes.bg_indexed = nil, nil
      pcall(vim.api.nvim_set_hl, 0, name, attributes)
    end
  end
end

function M.setup()
  M.apply()
  local group = vim.api.nvim_create_augroup('ModelMavenStandaloneNoColor', {clear = true})
  vim.api.nvim_create_autocmd('ColorScheme', {group = group, callback = function()
    vim.schedule(M.apply)
  end})
end

return M
