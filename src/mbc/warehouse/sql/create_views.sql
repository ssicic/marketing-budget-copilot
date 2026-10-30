-- What the MMM is fitted on: weekly sales plus a linear trend from 0 (first week)
-- to 1 (last week). PyMC-Marketing doesn't scale controls, so the trend stays on [0, 1].
CREATE OR REPLACE VIEW mmm_training_frame AS
SELECT
    date,
    spend_tv,
    spend_search,
    spend_social,
    spend_radio,
    promo,
    black_friday,
    christmas,
    (row_number() OVER (ORDER BY date) - 1) / (count(*) OVER () - 1) AS trend,
    sales
FROM weekly_sales;
