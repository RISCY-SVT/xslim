# Upstream base decision

The release candidate uses the required immutable upstream base
`9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`. The remotely observed upstream
`main` remained at that commit, so no newer-main scout or source substitution
was needed.

The two-input ReduceMax repair already present at this base is attributed to
upstream. The RISCY-SVT delta adds edge-semantics coverage and one independently
discovered empty-reduction correction; it does not claim authorship of the
upstream repair.
