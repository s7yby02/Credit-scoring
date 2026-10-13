select
    application_id,
    defaulted,
    outcome_date
from {{ source('raw', 'outcomes') }}
