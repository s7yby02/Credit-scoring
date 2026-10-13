-- Features + label, only for applications whose outcome is known
select
    f.*,
    o.defaulted
from {{ ref('fct_applicant_features') }} as f
inner join {{ ref('stg_outcomes') }} as o using (application_id)
