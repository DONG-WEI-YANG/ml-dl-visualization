# Curriculum quiz seed

`seed_v1.json` is the bundled baseline (version `2026-09-20.1`): 54 single-choice concept questions, three for each week 1–18. These are authored from the repository's weekly lectures, with each question carrying its source path. Weeks 3–4 also adapt concepts from the existing assessment template. This is a short weekly concept check, not the template's complete 100-point assessment. It includes no accounts, student records or learning events.

`seed_quiz_questions(conn)` validates the entire bundle before writing and inserts only missing stable IDs. It does not commit or close the connection. Initialization may call it inside its own transaction. Re-running does not overwrite teacher edits, custom questions, or conflicting IDs. Deleting a bundled question allows the next initialization to restore that missing ID; use an edited question to replace baseline content. Bundle version changes must preserve IDs for existing concepts; applying revised wording to an existing database is a separate explicit editorial operation.

Validation requires all 18 weeks, at least three questions per week, unique IDs/options, nonempty wording/explanations, valid integer answers and matching curriculum source paths. CI additionally checks that source files exist, all weeks serve questions without answers, and their stored answers grade to 100%. Startup validation checks the bundle, not editorial changes already stored in the database.

Run from `platform/backend`:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_quiz_seed.py tests/test_api_quiz.py tests/test_quiz_integrity.py -q
```

After deployment, verify public `/api/quiz/week/1` through `/api/quiz/week/18` each return at least three questions. Inspect answer validity through authorized local/admin review; public responses intentionally omit answers. No production writes or deployment are performed by these tests.
