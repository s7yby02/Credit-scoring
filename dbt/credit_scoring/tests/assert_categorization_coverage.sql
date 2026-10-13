-- Fails if more than 5% of transactions fall into 'other':
-- a sign that banks changed their description formats and the rules need updating.
select share_other
from (
    select countif(category = 'other') / count(*) as share_other
    from {{ ref('stg_transactions') }}
)
where share_other > 0.05
