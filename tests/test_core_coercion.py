"""Direct unit tests for the shared fail-closed coercion helpers in core.py.

``coerce_record_id`` / ``coerce_flag`` back every row-serving path and every
adapter's id/flag reads; this module tests them in isolation so the matrix of
accepted/refused shapes is pinned down independent of any one call site.

The two helpers share the numeric coercion body but differ in domain:
``coerce_record_id`` accepts any integral value, while ``coerce_flag`` accepts
**only** ``0``/``1`` (Fix 1a -- an out-of-domain flag must refuse, not leak).
The shared behavior class therefore uses only flag-domain-safe values; record
-id-specific and flag-specific cases live in their own sections below.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from jaenys.core import RedactionDriftError, coerce_flag, coerce_record_id


@pytest.mark.parametrize("coerce", [coerce_record_id, coerce_flag])
class TestSharedCoercionBehavior:
    """Behavior identical for both helpers, exercised with flag-domain-safe
    values (``0``/``1`` and their integral spellings) so the same accept cases
    hold for ``coerce_flag`` after its ``{0, 1}`` restriction."""

    def test_accepts_plain_int(self, coerce) -> None:
        assert coerce(1) == 1
        assert coerce(0) == 0

    def test_accepts_digit_string(self, coerce) -> None:
        assert coerce("1") == 1
        assert coerce("0") == 0

    def test_accepts_integral_float(self, coerce) -> None:
        assert coerce(1.0) == 1

    def test_accepts_integral_decimal(self, coerce) -> None:
        assert coerce(Decimal("1")) == 1

    def test_rejects_none(self, coerce) -> None:
        with pytest.raises(RedactionDriftError, match="missing"):
            coerce(None)

    def test_rejects_non_integral_float(self, coerce) -> None:
        with pytest.raises(RedactionDriftError, match="not an integral value"):
            coerce(3.7)

    def test_rejects_non_integral_decimal(self, coerce) -> None:
        with pytest.raises(RedactionDriftError, match="not an integral value"):
            coerce(Decimal("3.5"))

    def test_never_truncates(self, coerce) -> None:
        # A truncated id could mis-classify a span member -- confirm refusal,
        # not a silently truncated 3.
        with pytest.raises(RedactionDriftError):
            coerce(3.999999)

    def test_rejects_junk_string(self, coerce) -> None:
        with pytest.raises(RedactionDriftError, match="not an integer"):
            coerce("abc")

    def test_rejects_infinite_and_nan_floats(self, coerce) -> None:
        # int(inf) raises OverflowError, int(nan) raises ValueError -- both
        # must surface as the fail-closed error, never escape raw.
        with pytest.raises(RedactionDriftError, match="not an integer"):
            coerce(float("inf"))
        with pytest.raises(RedactionDriftError, match="not an integer"):
            coerce(float("nan"))

    def test_rejects_empty_string(self, coerce) -> None:
        with pytest.raises(RedactionDriftError):
            coerce("")

    def test_message_names_offending_value_type(self, coerce) -> None:
        with pytest.raises(RedactionDriftError, match=r"<str>"):
            coerce("abc")

    def test_message_names_origin_when_given(self, coerce) -> None:
        with pytest.raises(RedactionDriftError, match="my-store"):
            coerce("abc", origin="my-store")

    def test_omitted_origin_produces_no_at_clause(self, coerce) -> None:
        with pytest.raises(RedactionDriftError) as exc_info:
            coerce(None)
        assert " at " not in str(exc_info.value)


class TestRecordIdSpecificAccepts:
    """Values outside the flag domain that ``coerce_record_id`` still accepts --
    any integral id is legal, so these must not be folded into the shared
    (flag-domain-safe) accept cases above."""

    def test_accepts_arbitrary_int(self) -> None:
        assert coerce_record_id(7) == 7

    def test_accepts_arbitrary_digit_string(self) -> None:
        assert coerce_record_id("42") == 42

    def test_accepts_arbitrary_integral_float(self) -> None:
        assert coerce_record_id(3.0) == 3

    def test_accepts_arbitrary_integral_decimal(self) -> None:
        assert coerce_record_id(Decimal("3")) == 3

    def test_rejects_bool(self) -> None:
        with pytest.raises(RedactionDriftError, match="record id is not an integer"):
            coerce_record_id(True)


class TestFlagDomainRejections:
    """``coerce_flag`` refuses any integral value outside ``{0, 1}`` (Fix 1a);
    the same values are perfectly valid record ids."""

    @pytest.mark.parametrize("value", [2, -1, "7", 42, Decimal("2"), 2.0])
    def test_rejects_out_of_domain(self, value) -> None:
        with pytest.raises(RedactionDriftError, match="flag must be 0 or 1"):
            coerce_flag(value)
        # The same value is a legal record id -- proving the restriction is
        # flag-specific, not a shared numeric refusal.
        coerce_record_id(value)

    def test_out_of_domain_message_names_value_and_origin(self) -> None:
        with pytest.raises(RedactionDriftError, match=r"flag must be 0 or 1 at my-store"):
            coerce_flag(2, origin="my-store")

    def test_accepts_bool(self) -> None:
        assert coerce_flag(True) == 1
        assert coerce_flag(False) == 0


@pytest.mark.parametrize("bad", ["1_0", " 7 ", "+7", "٧"])
def test_rejects_int_parseable_but_unsafe_digit_strings(bad: str) -> None:
    # int() accepts these and would remap a corrupt id to a different record;
    # only a bare optional-sign ASCII digit run is allowed.
    with pytest.raises(RedactionDriftError, match="not an integer"):
        coerce_record_id(bad)


def test_underscored_flag_string_refuses() -> None:
    # "0_0" would otherwise int() to 0 and serve a corrupt row clear.
    with pytest.raises(RedactionDriftError, match="not an integer"):
        coerce_flag("0_0")


@pytest.mark.parametrize(("value", "expected"), [("42", 42), ("-3", -3), ("07", 7)])
def test_bare_digit_strings_still_coerce(value: str, expected: int) -> None:
    assert coerce_record_id(value) == expected


def test_coerce_record_id_message_mentions_record_id() -> None:
    with pytest.raises(RedactionDriftError, match="record id"):
        coerce_record_id(None)


def test_coerce_flag_message_mentions_flag() -> None:
    with pytest.raises(RedactionDriftError, match="flag"):
        coerce_flag(None)


def test_refusal_does_not_echo_corrupt_value() -> None:
    junk = "patient-SSN-123456789-CONFIDENTIAL"
    with pytest.raises(RedactionDriftError) as exc_info:
        coerce_record_id(junk)
    message = str(exc_info.value)
    assert junk not in message
    assert "<str>" in message


def test_refusal_reports_value_type_only() -> None:
    with pytest.raises(RedactionDriftError, match=r"<str>"):
        coerce_record_id("abc")
    with pytest.raises(RedactionDriftError, match=r"<float>"):
        coerce_record_id(3.7)
