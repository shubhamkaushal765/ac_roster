// API Response wrapper
export interface ApiResponse<T> {
    success: boolean
    data?: T
    error?: {
        code: string
        message: string
        details?: never[]
    }
}

// Enums
export type OperationMode = "arrival" | "departure"
export type EditType = "delete" | "swap" | "add"

// Roster Generation
export interface RosterGenerationRequest {
    mode: OperationMode
    main_officers_reported: string
    report_gl_counters?: string
    handwritten_counters?: string
    ot_counters?: string
    ro_ra_officers?: string
    sos_timings?: string
    beam_width?: number
    save_to_history?: boolean
}

export interface OfficerCounts {
    main: number
    sos: number
    ot: number
    total: number
}

export interface StatisticsData {
    stats1: string
    stats2: string
}

export interface RosterData {
    officer_schedules: Record<string, number[]>
    counter_matrix: number[][]
    mode: OperationMode
}

export interface RosterGenerationResponse {
    success: boolean
    data: RosterData
    officer_counts: OfficerCounts
    optimization_penalty?: number
    statistics: StatisticsData
}

// History
export interface RosterHistoryItem {
    id: number
    timestamp: string
    main_officers: string
    gl_counters?: string
    handwritten_counters?: string
    ot_counters?: string
    ro_ra_officers?: string
    sos_timings?: string
    beam_width: number
    optimization_penalty?: number
    main_officer_count?: number
    sos_officer_count?: number
    ot_officer_count?: number
    total_officer_count?: number
    notes?: string
}

export interface RosterHistoryResponse {
    success: boolean
    data: RosterHistoryItem[]
    count: number
}

export interface LastInputsData {
    main_officers: string
    gl_counters: string
    handwritten_counters: string
    ot_counters: string
    ro_ra_officers: string
    sos_timings: string
    raw_sos_text: string
    beam_width: number
}

export interface LastInputsResponse {
    success: boolean
    data: LastInputsData
}

// Edits
export interface RosterEditCreate {
    edit_type: EditType
    officer_id: string
    officer_id_2?: string
    counter_no?: number
    slot_start: number
    slot_end: number
    time_start: string
    time_end: string
    roster_history_id?: number
    notes?: string
}

export interface RosterEdit extends RosterEditCreate {
    id: number
    timestamp: string
}

export interface RosterEditResponse {
    success: boolean
    edit_id: number
    message: string
}

export interface RosterEditsListResponse {
    success: boolean
    data: RosterEdit[]
    count: number
}

export interface SuccessResponse {
    success: boolean
    message: string
}