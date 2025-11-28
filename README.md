# ml_url_classification

### Mike Notes

Leave-One-User-Out CV too long for our dataset size of ~650,000 rows (found through trying it on a traditional logistic regression model). I ran it for 4 hours but to no avail.

## Current features: 15 so far
url_length
num_digits
num_periods
num_slashes
num_ats
min_brand_dist
is_typosquat
has_html
has_query_param
has_https
has_http
has_ip_address
has_suspicious_kw
has_brand_kw
has_non_ascii_chars
digit_len_ratio

## Dropped features: 5
url
malicious
type
domain
users - shouldn't this be url instead of users? we don't have users

## Target: 1
malicious