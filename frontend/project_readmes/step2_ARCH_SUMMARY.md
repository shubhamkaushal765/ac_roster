## Architecture Summary

### Running the application

```bash
# Start Next.js dev server
npm run dev

# Build for production
npm run build
npm start
```

### SSR Strategy

- **Dashboard (/):** Fetches recent history server-side
- **Generate (/generate):** Fetches last inputs server-side, form client-side
- **History (/history):** Full SSR for history list
- **Edits (/edits):** Hybrid – fetches edits server-side, form client-side

### Client Components Usage

- **GenerationForm:** Requires form state, validation, submission
- **EditForm:** Requires form state, validation, submission
- **ResultPage:** Uses sessionStorage (browser-only)

### Server Components

- **RosterDisplay:** Pure display, no interactivity
- **HistoryTable:** Pure display
- **EditsList:** Pure display
- All page data fetching

### Key Decisions

- No `useEffect` for initial data – all initial fetches in Server Components
- Client forms only – Server Actions not used (could be added)
- `sessionStorage` for results – avoids URL query params for large data
- Centralized API client – single source of truth for endpoints
- Zod validation – ensures type safety with backend schemas

### Production Checklist

- Set `NEXT_PUBLIC_API_URL` in production `.env`
- Add error boundaries for runtime errors
- Add loading states for page transitions
- Implement proper caching strategy (ISR/revalidation)
- Add authentication if required
- Add toast notifications for success/error
- Optimize table pagination for large datasets
- Add download/export functionality for rosters
- Add real-time updates (WebSocket) if needed  
