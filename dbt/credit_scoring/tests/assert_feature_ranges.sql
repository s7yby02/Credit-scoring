-- Returns the rows with impossible feature values (the test fails if any exist)
select *
from {{ ref('fct_applicant_features') }}
where avg_monthly_income < 0
    or income_cv < 0
    or rent_to_income < 0
    or gambling_share not between 0 and 1
    or negative_balance_days > 180
