select
    application_id,
    user_id,
    account_id,
    application_date,
    requested_amount,
    partner_id
from {{ source('raw', 'applications') }}
