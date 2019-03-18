workflow "Lint" {
  on = "push"
  resolves = ["Black Code Formatter", "Pyflakes Syntax Checker"]
}

action "Black Code Formatter" {
  uses = "lgeiger/black-action@v1.0.1"
  args = ". --check"
}

action "Pyflakes Syntax Checker" {
  uses = "lgeiger/pyflakes-action@v1.0.1"
}
