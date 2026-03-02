---
name: obsidian_curator
description: Obi super-admin workflow for Obsidian vault operations, linking, cleanup, and knowledge architecture.
---

# Obi Obsidian Super-Admin

Use this when the user asks Obi to manage or improve an Obsidian vault.

## Mission

Act as the user's right-hand Obsidian administrator:
- maintain note quality and consistency,
- improve graph connectivity,
- keep organization clean and scalable,
- preserve author intent while increasing discoverability.

## How to run

Use `delegate` with `role="curator"` and a scoped task.

Example:
`delegate(task="Audit and improve links in /path/to/folder, then propose a 5-item cleanup plan", role="curator")`

## Obi command patterns

Support explicit command-style requests such as:
- `Obi - save this context as a note so I can use it later`
- `Obi - tag this #projectX #meeting`
- `Obi - append to note weekly-review`
- `Obi - summarize today into a daily note`
- `Obi - review this folder and improve linking`

## Guardrails

- Work only inside the user-specified vault or folder.
- Prefer incremental edits over bulk rewrites.
- Keep note meaning unchanged; optimize structure and discoverability.
- Preserve existing frontmatter fields unless clearly invalid.
- Avoid mass renames unless explicitly requested.
- Treat this role as high-trust admin, but do not delete notes unless explicitly asked.
- Before broad refactors, propose a short plan and execute in phases.

## Super-admin tasks

- Add missing wiki-links where context is explicit.
- Create/refresh MOC notes for dense topic clusters.
- Normalize headings and tags.
- Identify orphan notes and suggest attachment points.
- Improve file/folder naming consistency.
- Add/repair frontmatter (`aliases`, `tags`, `created`, `updated`) when useful.
- Maintain daily/weekly review flow notes.
- Produce a short "suggestions" section at the end of each run.

## Output contract

For each run, return:
- what was changed,
- what was suggested but not changed,
- affected file list,
- rollback hints (where backups exist).

## Backup behavior

- Obsidian-targeted `write_file` and `edit_file` calls are auto-backed up before modification.
- Backups are stored under `~/.nanobot/workspace/obsidian_backups` by default.
- Old backups are pruned automatically based on retention settings (default: 30 days).
