# Example: why this shipment was flagged

Predicted miss probability: **20%**

| Driver | Value | Contribution to risk (log-odds) |
|---|---|---|
| dest_region_northeast | 1 | +0.38 (raises risk) |
| distance_miles | 1.43e+03 | +0.24 (raises risk) |
| day_of_week | 0 | +0.08 (raises risk) |
| route_stop_density | 0.05 | +0.07 (raises risk) |
| minutes_after_cutoff | 657 | +0.07 (raises risk) |
| package_weight_lb | 26.5 | +0.06 (raises risk) |
| is_rural_dest | 1 | -0.04 (lowers risk) |
| miles_per_promised_day | 143 | +0.04 (raises risk) |

_Positive contributions push toward a missed commitment; negative pull away._