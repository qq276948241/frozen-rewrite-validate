"""
Consistency validation across the five cooperating features: `evolve`,
`asdict`, `fields`, slots, and default factories.
"""

import pytest

import attr
from attr import asdict, define, evolve, field, fields, frozen
from attr.exceptions import NotAnAttrsClassError


@define
class TwoFields:
    """A plain two-field class without slots tricks."""

    a: int
    b: int = 2


@frozen
class FrozenPoint:
    x: int
    y: int = 2


@define
class Converted:
    x: int = field(converter=int, validator=attr.validators.gt(0))


@define
class WithNonCompare:
    a: int
    b: int = field(eq=False)


@frozen
class HashableNonCompare:
    a: int
    b: int = field(eq=False)


@define
class WithFactory:
    items: list = field(factory=list)


@define
class Base:
    a: int


@define
class Sub(Base):
    b: int = 0


@define
class Private:
    _x: int
    y: int = 1


class TestEvolve:
    def test_only_named_fields_change(self):
        """Unmentioned fields keep their old values; the new instance is a
        distinct object."""
        old = TwoFields(1, 2)
        new = evolve(old, a=10)

        assert new is not old
        assert (new.a, new.b) == (10, 2)
        assert (old.a, old.b) == (1, 2)

    def test_frozen_class_can_evolve(self):
        p = FrozenPoint(1)
        p2 = evolve(p, x=10)

        assert p2 is not p
        assert (p2.x, p2.y) == (10, 2)
        assert (p.x, p.y) == (1, 2)

    def test_new_values_are_converted_and_validated(self):
        c = Converted(5)

        assert evolve(c, x="7").x == 7

        with pytest.raises(Exception):
            evolve(c, x=-3)

        assert c.x == 5

    def test_failed_evolve_leaves_old_instance_untouched(self):
        c = Converted(5)
        snapshot = asdict(c)

        with pytest.raises(Exception):
            evolve(c, x=-1)

        assert asdict(c) == snapshot
        assert c.x == 5

    def test_unknown_field_is_rejected(self):
        old = TwoFields(1, 2)

        with pytest.raises(TypeError):
            evolve(old, nope=1)

        assert (old.a, old.b) == (1, 2)

    def test_chained_evolve_builds_on_previous(self):
        p = FrozenPoint(1)
        p2 = evolve(evolve(p, x=5), y=9)

        assert (p2.x, p2.y) == (5, 9)
        assert (p.x, p.y) == (1, 2)

    def test_subclass_fields_survive_evolve(self):
        s = Sub(1, 2)
        s2 = evolve(s, a=9)

        assert type(s2) is Sub
        assert (s2.a, s2.b) == (9, 2)

    def test_evolve_does_not_inject_subclass_fields_into_base(self):
        b = Base(1)
        b2 = evolve(b, a=3)

        assert type(b2) is Base
        assert not hasattr(b2, "b")
        assert [f.name for f in fields(Base)] == ["a"]

    def test_evolve_requires_attrs_instance(self):
        with pytest.raises(NotAnAttrsClassError):
            evolve(object())


class TestAsdict:
    def test_non_compare_fields_included_by_default(self):
        assert asdict(WithNonCompare(1, 2)) == {"a": 1, "b": 2}

    def test_filter_drops_only_requested(self):
        d = WithNonCompare(1, 2)
        filtered = asdict(d, filter=lambda a, v: a.name != "b")

        assert filtered == {"a": 1}
        assert list(filtered) == ["a"]

    def test_key_order_matches_definition_order(self):
        @define
        class Ordered:
            z: int = 0
            m: int = 0
            a: int = 0

        assert list(asdict(Ordered())) == ["z", "m", "a"]

    def test_retain_collection_types_keeps_values(self):
        @define
        class Holder:
            items: list = field(factory=list)

        h = Holder([1, 2])

        plain = asdict(h)
        retained = asdict(h, retain_collection_types=True)

        assert plain["items"] == [1, 2]
        assert retained["items"] == [1, 2]

    def test_slots_off_extra_attributes_not_fields_by_default(self):
        @define(slots=False)
        class Loose:
            a: int

        loose = Loose(1)
        loose.extra = 5

        assert loose.extra == 5
        assert asdict(loose) == {"a": 1}

    def test_dropped_field_absent_from_mapping(self):
        d = asdict(TwoFields(1, 2), filter=lambda a, v: a.name != "b")

        assert "b" not in d
        assert d["a"] == 1


class TestFields:
    def test_order_matches_definition(self):
        assert [f.name for f in fields(WithNonCompare)] == ["a", "b"]

    def test_private_names_are_the_defined_names(self):
        """The generated init alias never leaks as the field name."""
        names = [f.name for f in fields(Private)]

        assert names == ["_x", "y"]
        assert [f.alias for f in fields(Private)] == ["x", "y"]

    def test_fields_of_plain_two_field_class(self):
        assert [f.name for f in fields(TwoFields)] == ["a", "b"]


class TestSlots:
    def test_slotted_instance_rejects_new_attribute(self):
        t = TwoFields(1, 2)

        with pytest.raises(AttributeError):
            t.zzz = 1

    def test_rejection_happens_at_assignment_time(self):
        t = TwoFields(1, 2)

        try:
            t.new_name = 1
        except AttributeError:
            pass
        else:
            pytest.fail("slotted instance accepted a new attribute")

        assert not hasattr(t, "new_name")

    def test_slots_off_allows_extra_attributes(self):
        @define(slots=False)
        class Loose:
            a: int

        loose = Loose(1)
        loose.anything = 42

        assert loose.anything == 42


class TestFactory:
    def test_each_instance_gets_a_fresh_container(self):
        f1, f2 = WithFactory(), WithFactory()

        f1.items.append(1)

        assert f2.items == []
        assert f1.items is not f2.items

    def test_factory_failure_leaves_no_half_instance(self):
        @define
        class Exploding:
            items: list = field(factory=lambda: 1 / 0)

        with pytest.raises(ZeroDivisionError):
            Exploding()


class TestCrossFeatureConsistency:
    def test_eq_hash_asdict_agree_on_non_compare_field(self):
        """eq and hash ignore the field; asdict still keeps it."""
        h1, h2 = HashableNonCompare(1, 2), HashableNonCompare(1, 9)

        assert h1 == h2
        assert hash(h1) == hash(h2)
        assert asdict(h1) == {"a": 1, "b": 2}
        assert asdict(h2) == {"a": 1, "b": 9}

    def test_evolve_then_asdict_round_trip(self):
        old = TwoFields(1, 2)
        new = evolve(old, a=10)

        assert asdict(new) == {"a": 10, "b": 2}
        assert asdict(old) == {"a": 1, "b": 2}

    def test_evolve_uses_fields_list_order_and_names(self):
        """Every init field reported by `fields` is carried over by evolve."""
        p = Private(1, 2)
        p2 = evolve(p, y=9)

        assert asdict(p2) == {"_x": 1, "y": 9}

    def test_frozen_slotted_factory_all_together(self):
        @frozen
        class AllTogether:
            name: str = field(validator=attr.validators.instance_of(str))
            tags: list = field(factory=list)
            cache: int = field(default=0, eq=False)

        first = AllTogether("a")
        second = AllTogether("b")
        first.tags.append("x")

        assert second.tags == []

        evolved = evolve(first, name="c")
        assert evolved is not first
        assert evolved.name == "c"
        assert evolved.tags == ["x"]
        assert asdict(evolved) == {"name": "c", "tags": ["x"], "cache": 0}

        with pytest.raises(AttributeError):
            evolved.other = 1

        with pytest.raises(Exception):
            evolve(first, name=123)

        assert first.name == "a"
        assert first.tags == ["x"]

    def test_failure_sources_are_distinguishable(self):
        """Each of the five features fails in an identifiable way."""
        with pytest.raises(TypeError):
            evolve(TwoFields(1, 2), unknown=1)  # evolve

        with pytest.raises(NotAnAttrsClassError):
            asdict(object())  # asdict

        with pytest.raises(NotAnAttrsClassError):
            fields(object)  # fields

        with pytest.raises(AttributeError):
            TwoFields(1, 2).rogue = 1  # slots

        @define
        class Exploding:
            items: list = field(factory=lambda: 1 / 0)

        with pytest.raises(ZeroDivisionError):
            Exploding()  # factory
