-- AI VOC Ops v0.3 pilot feedback table
-- Run this in a dedicated Supabase project.

create table if not exists public.pilot_feedback (
    feedback_id uuid primary key,
    created_at timestamptz not null,
    session_id text not null,
    message_redacted text not null,
    message_fingerprint text not null,
    prediction_category text not null,
    prediction_subcategory text not null,
    prediction_priority text not null,
    prediction_sentiment text not null,
    prediction_human_review boolean not null,
    prediction_reason text not null,
    is_correct boolean not null,
    corrected_category text not null,
    corrected_subcategory text not null,
    corrected_priority text not null,
    corrected_sentiment text not null,
    corrected_human_review boolean not null,
    feedback_note text not null default '',
    model text,
    input_tokens bigint not null default 0,
    cached_input_tokens bigint not null default 0,
    output_tokens bigint not null default 0,
    estimated_cost_usd double precision,
    prompt_version text not null,
    taxonomy_version text not null
);

alter table public.pilot_feedback enable row level security;

revoke all on table public.pilot_feedback from anon;
grant insert on table public.pilot_feedback to anon;

drop policy if exists pilot_feedback_anon_insert on public.pilot_feedback;
create policy pilot_feedback_anon_insert
on public.pilot_feedback
for insert
to anon
with check (true);

-- Intentionally no anon SELECT/UPDATE/DELETE policy.
-- Use the Supabase dashboard or a trusted local service-role key for export/review.
