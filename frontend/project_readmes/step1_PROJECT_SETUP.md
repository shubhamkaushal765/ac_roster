# Create Next.js project with TypeScript

```
npx create-next-app@latest roster-frontend --typescript --tailwind --app --no-src-dir --import-alias "@/*"

cd roster-frontend
```

# Install dependencies

```
npm install @hookform/resolvers zod react-hook-form
npm install date-fns
```

# Install shadcn/ui

```
npx shadcn@latest init -d
```

# Install required shadcn components

```
npx shadcn@latest add button
npx shadcn@latest add input
npx shadcn@latest add label
npx shadcn@latest add card
npx shadcn@latest add select
npx shadcn@latest add table
npx shadcn@latest add tabs
npx shadcn@latest add textarea
npx shadcn@latest add badge
npx shadcn@latest add alert
npx shadcn@latest add skeleton
npx shadcn@latest add switch
npx shadcn@latest add dialog
npx shadcn@latest add separator
```

---

## Step 2: Project Structure

```
roster-frontend/
├── app/
│   ├── layout.tsx                 # Root layout
│   ├── page.tsx                   # Home/Dashboard
│   ├── generate/
│   │   └── page.tsx              # Roster generation (SSR + Client form)
│   ├── history/
│   │   └── page.tsx              # History list (SSR)
│   ├── edits/
│   │   └── page.tsx              # Edits management (SSR)
│   └── api/                      # (unused, all API calls to FastAPI)
├── components/
│   ├── roster/
│   │   ├── generation-form.tsx   # Client Component (form)
│   │   ├── roster-display.tsx    # Server Component (display)
│   │   └── roster-statistics.tsx # Server Component
│   ├── history/
│   │   └── history-table.tsx     # Server Component
│   ├── edits/
│   │   ├── edit-form.tsx         # Client Component
│   │   └── edits-list.tsx        # Server Component
│   └── ui/                       # shadcn components
├── lib/
│   ├── api.ts                    # Centralized API client
│   ├── types.ts                  # TypeScript types from backend
│   ├── utils.ts                  # Utility functions
│   └── validations.ts            # Zod schemas
├── .env.local
└── next.config.js