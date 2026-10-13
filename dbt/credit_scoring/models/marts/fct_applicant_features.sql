-- One row of features per application, using only data known at application time.
-- Mirrors features/compute.py (checked by tests/test_parity.py).

with applications as (
    select * from {{ ref('stg_applications') }}
),

-- point-in-time window: the 180 days strictly before the application date
windowed as (
    select
        a.application_id,
        t.booking_date,
        t.amount,
        t.category,
        t.balance_after,
        div(date_diff(a.application_date, t.booking_date, day) - 1, 30) as month_bucket  -- 0 = most recent
    from applications as a
    inner join {{ ref('stg_transactions') }} as t
        on t.account_id = a.account_id
        and t.booking_date < a.application_date
        and t.booking_date >= date_sub(a.application_date, interval 180 day)
),

-- 6 monthly buckets per application, so months without income count as 0
months as (
    select a.application_id, bucket
    from applications as a
    cross join unnest(generate_array(0, 5)) as bucket
),

monthly_income as (
    select
        m.application_id,
        m.bucket,
        coalesce(sum(if(w.category in ('salary', 'freelance_income', 'benefits'), w.amount, 0)), 0) as income
    from months as m
    left join windowed as w
        on w.application_id = m.application_id
        and w.month_bucket = m.bucket
    group by 1, 2
),

income_stats as (
    select
        application_id,
        avg(income) as avg_monthly_income,
        safe_divide(stddev_pop(income), avg(income)) as income_cv
    from monthly_income
    group by 1
),

spend_stats as (
    select
        application_id,
        sum(if(category = 'rent', -amount, 0)) / 6 as avg_monthly_rent,
        sum(if(amount < 0, -amount, 0)) as total_spend,
        sum(if(category = 'gambling', -amount, 0)) as gambling_spend,
        countif(category = 'rejected_debit') as n_rejected_debits,
        count(distinct if(balance_after < 0, booking_date, null)) as negative_balance_days,
        sum(if(amount > 0, amount, 0)) as total_inflows,
        sum(amount) as net_cashflow,
        count(*) as n_transactions
    from windowed
    group by 1
)

select
    a.application_id,
    a.user_id,
    a.account_id,
    a.application_date,
    coalesce(s.n_transactions, 0) as n_transactions,
    i.avg_monthly_income,
    i.income_cv,
    safe_divide(s.avg_monthly_rent, i.avg_monthly_income) as rent_to_income,
    safe_divide(s.gambling_spend, s.total_spend) as gambling_share,
    coalesce(s.n_rejected_debits, 0) as n_rejected_debits,
    coalesce(s.negative_balance_days, 0) as negative_balance_days,
    safe_divide(s.net_cashflow, s.total_inflows) as net_cashflow_ratio
from applications as a
left join income_stats as i using (application_id)
left join spend_stats as s using (application_id)
