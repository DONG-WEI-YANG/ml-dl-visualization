# ML/DL Visualization Frontend

React + TypeScript interactive teaching platform for ML/DL visualization.

## Requirements

- Node.js 18+
- npm

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Start dev server (auto-proxies API to localhost:8000)
npm run dev

# 3. Open http://localhost:5173
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Type-check and build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm test` | Run unit tests (Vitest) |
| `npm run test:watch` | Run tests in watch mode |

## Environment Variables

Create `.env` from `.env.example`:

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE` | Backend API URL (leave empty for Vite proxy) | `""` |

## Project Structure

```
frontend/
  src/
    App.tsx                  # Router + ErrorBoundary
    main.tsx                 # Entry point
    components/
      Layout.tsx             # Page layout with sidebar
      Sidebar.tsx            # Navigation sidebar
      ErrorBoundary.tsx      # Error boundary wrapper
      llm/ChatPanel.tsx      # AI tutor chat interface
      quiz/QuizPanel.tsx     # Weekly quiz component
      viz/                   # 17 visualization components
        ActivationFunctionViz.tsx   # W11: Activation functions
        AttentionViz.tsx            # W13: Attention heatmap
        CNNLayerViz.tsx             # W12: CNN convolution
        DataSplitViz.tsx            # W3: Train/val/test split
        DecisionBoundaryViz.tsx     # W5-6: Decision boundary
        EDAViz.tsx                  # W2: Exploratory data analysis
        EmbeddingSpaceViz.tsx       # W17: Embedding visualization
        EnvironmentSetupViz.tsx     # W1: Setup checklist
        FairnessViz.tsx             # W15: Fairness metrics
        FeatureImportanceViz.tsx    # W8: SHAP values
        GradientDescentViz.tsx      # W4: Gradient descent
        LearningCurveViz.tsx        # W10: Learning curves
        MLOpsFlowViz.tsx            # W16: MLOps pipeline
        PipelineFlowViz.tsx         # W9: Data pipeline
        ProjectShowcaseViz.tsx      # W18: Project showcase
        TrainingComparisonViz.tsx   # W14: Training comparison
        TreeGrowthViz.tsx           # W7: Decision tree
    pages/
      Home.tsx               # Week grid overview
      WeekPage.tsx           # Individual week page
      Dashboard.tsx          # Learning analytics
      AdminSettings.tsx      # Admin panel
      NotFound.tsx           # 404 page
    hooks/
      useChat.ts             # WebSocket chat hook
    lib/
      api.ts                 # API fetch utilities
    types.ts                 # Shared types
  test/                      # Vitest tests
```

## Tech Stack

- **React 19** + **TypeScript**
- **Vite 6** (bundler)
- **Tailwind CSS 4** (styling)
- **Recharts** + **D3.js** (visualization)
- **React Router 7** (routing)
- **Vitest** + **Testing Library** (testing)

## Docker

```bash
docker build -t ml-dl-frontend .
docker run -p 3000:80 ml-dl-frontend
```
