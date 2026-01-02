def main [...args] {
  let repo = ($env.FILE_PWD | path dirname)
  ^uv run --directory $repo python scripts/paper_figs.py ...$args
}
