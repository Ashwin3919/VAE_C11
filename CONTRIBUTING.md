# Contributing to C-VAE

Thank you for contributing. This document covers commit conventions, branch naming, and the PR checklist.

## Commit Format (Conventional Commits)

Every commit title must follow the [Conventional Commits](https://www.conventionalcommits.org/) spec:

```
<type>(<scope>): <short summary>
```

| Type       | Use for                                        |
|------------|------------------------------------------------|
| `feat`     | New feature or behaviour                       |
| `fix`      | Bug fix                                        |
| `perf`     | Performance improvement (no behaviour change)  |
| `refactor` | Code restructuring (no feature / bug change)   |
| `test`     | Adding or updating tests                       |
| `docs`     | Documentation only                             |
| `build`    | Makefile, CI, or toolchain changes             |
| `chore`    | Housekeeping (formatting, renaming, etc.)      |

**Scope** is optional but encouraged: `forward`, `backward`, `loss`, `optimizer`, `loader`, `train`, `ci`.

### Good examples
```
feat(loader): accept runtime --digits filter, remove #ifdef FULL_MNIST
perf(forward): tile GEMM 64×64 to fit L1 cache
fix(backward): restore dec_b3[k] after numerical gradient check
test(model): add fuzz test covering 50 random forward pass inputs
build(ci): add GitHub Actions tsan compilation check
```

### Bad examples (avoid)
```
fixes
wokring not best veriosn yet
updates
```

## Branch Naming

```
<type>/<short-description>
```

Examples: `feat/runtime-digit-filter`, `fix/slab-mismatch`, `perf/avx2-gemm`

## Pull Request Checklist

Before opening a PR, confirm:

- [ ] `make test` passes locally
- [ ] `make asan` compiles without errors
- [ ] New behaviour is covered by a test in `tests/`
- [ ] Commit titles follow Conventional Commits
- [ ] No typos in commit messages (spell-check your subject line)

## Code Style

- C11, `-Wall -Wextra -Wpedantic` clean
- `snake_case` for functions and variables
- Document non-obvious invariants with inline comments
- Prefer runtime guards over `assert()` (assert is compiled out with `-DNDEBUG`)
