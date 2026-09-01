# Score Boosting - Qdrant

## Linear Decay example

### We're essentially onstructing a Qdrant expression using Qdrant's API objects:

```bash
models.FormulaQuery
       ↓
models.SumExpression
       ↓
models.MultExpression
       ↓
models.LinDecayExpression
       ↓
models.DecayParamsExpression
       ├── x
       ├── target
       ├── scale
  `    └── midpoint

| Parameter  | Qdrant API meaning                                  | Yours                   |
| ---------- | --------------------------------------------------- | ----------------------- |
| `x`        | The value to apply the decay to                     | `"published_timestamp"` |
| `target`   | The ideal/reference value                           | `now`                   |
| `scale`    | Distance over which the decay occurs                | `30 days`               |
| `midpoint` | Controls where the decay reaches the midpoint value | `0.3`                   |
`
