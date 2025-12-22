import {
    ApiResponse,
    RosterGenerationRequest,
    RosterGenerationResponse,
    RosterHistoryResponse,
    LastInputsResponse,
    RosterEditCreate,
    RosterEditResponse,
    RosterEditsListResponse,
    SuccessResponse,
    LastInputsData,
} from "./types"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const API_V1 = `${API_URL}/api/v1`

class ApiError extends Error {
    constructor(
        public status: number,
        public code: string,
        message: string,
        public details?: never[]
    ) {
        super(message)
        this.name = "ApiError"
    }
}

async function fetchApi<T>(
    endpoint: string,
    options?: RequestInit
): Promise<T> {
    const response = await fetch(`${API_V1}${endpoint}`, {
        headers: {
            "Content-Type": "application/json",
            ...options?.headers,
        },
        ...options,
    })

    const data = await response.json()

    if (!response.ok || !data.success) {
        throw new ApiError(
            response.status,
            data.error?.code || "UNKNOWN_ERROR",
            data.error?.message || "An error occurred",
            data.error?.details
        )
    }

    return data
}

// Roster Generation
export async function generateRoster(
    request: RosterGenerationRequest
): Promise<RosterGenerationResponse> {
    return fetchApi<RosterGenerationResponse>("/roster/generate", {
        method: "POST",
        body: JSON.stringify(request),
    })
}

// History
export async function getLastInputs(): Promise<LastInputsResponse> {
    return fetchApi<LastInputsResponse>("/history/last-inputs")
}

export async function saveLastInputs(
    inputs: Partial<LastInputsData>
): Promise<SuccessResponse> {
    return fetchApi<SuccessResponse>("/history/last-inputs", {
        method: "POST",
        body: JSON.stringify(inputs),
    })
}

export async function getRosterHistory(
    limit = 10
): Promise<RosterHistoryResponse> {
    return fetchApi<RosterHistoryResponse>(
        `/history/history?limit=${limit}`
    )
}

// Edits
export async function createRosterEdit(
    edit: RosterEditCreate
): Promise<RosterEditResponse> {
    return fetchApi<RosterEditResponse>("/edits/", {
        method: "POST",
        body: JSON.stringify(edit),
    })
}

export async function getRosterEdits(
    limit = 20
): Promise<RosterEditsListResponse> {
    return fetchApi<RosterEditsListResponse>(`/edits/?limit=${limit}`)
}

export async function deleteRosterEdit(
    editId: number
): Promise<SuccessResponse> {
    return fetchApi<SuccessResponse>(`/edits/${editId}`, {
        method: "DELETE",
    })
}

export async function clearAllRosterEdits(): Promise<SuccessResponse> {
    return fetchApi<SuccessResponse>("/edits/", {
        method: "DELETE",
    })
}

export {ApiError}