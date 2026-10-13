{{ config(materialized='table', cluster_by=['account_id']) }}

with transactions as (
    select
        transaction_id,
        account_id,
        user_id,
        booked_at,
        booking_date,
        amount,
        upper(description) as description,
        balance_after
    from {{ source('raw', 'transactions') }}
    where true
    -- defensive dedup: keep the latest load of each transaction
    qualify row_number() over (partition by transaction_id order by _loaded_at desc) = 1
),

rules as (
    select * from {{ ref('category_rules') }}
),

-- first matching rule wins (lowest priority number), as in features/compute.py
matched as (
    select
        t.transaction_id,
        r.category,
        row_number() over (partition by t.transaction_id order by r.priority) as rule_rank
    from transactions as t
    inner join rules as r
        on regexp_contains(t.description, r.pattern)
        and (
            r.direction = 'any'
            or (r.direction = 'credit' and t.amount > 0)
            or (r.direction = 'debit' and t.amount < 0)
        )
)

select
    t.*,
    coalesce(m.category, 'other') as category
from transactions as t
left join matched as m
    on t.transaction_id = m.transaction_id
    and m.rule_rank = 1
