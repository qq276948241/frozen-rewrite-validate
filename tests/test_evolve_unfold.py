"""
Consistency tests for the five features that must line up:

- `attrs.evolve` -- derive a new instance from an old one
- `attrs.asdict` / `attr.asdict` -- unfold an instance into a plain mapping
- `attrs.fields` -- the field roster
- slotted classes -- slot storage
- `attrs.Factory` -- default factories
"""

import pytest

import attr
import attrs
from attrs.exceptions import FrozenInstanceError


class TestEvolve:
    def test_changes_only_named_fields(self):
        """
        Evolving changes only the named fields; every other field keeps
        its value, and the result is a new object.
        """

        @attrs.define
        class C:
            a: int
            b: int

        old = C(1, 2)
        new = attrs.evolve(old, a=10)

        assert new is not old
        assert (new.a, new.b) == (10, 2)
        assert (old.a, old.b) == (1, 2)

    def test_frozen_class_evolves(self):
        """
        Frozen classes can be evolved; frozen-ness must not prevent
        creating a new instance.
        """

        @attrs.frozen
        class C:
            x: int
            y: int

        old = C(1, 2)
        new = attrs.evolve(old, x=5)

        assert (new.x, new.y) == (5, 2)
        assert (old.x, old.y) == (1, 2)
        with pytest.raises(FrozenInstanceError):
            old.x = 99

    def test_new_values_are_converted_and_validated(self):
        """
        Values passed to evolve go through converters and validators,
        also on frozen classes.
        """

        @attrs.frozen
        class C:
            x: int = attrs.field(converter=int)
            y: int = attrs.field(validator=attrs.validators.gt(0))

        inst = C(1, 2)
        new = attrs.evolve(inst, x="42")

        assert new.x == 42
        assert isinstance(new.x, int)
        with pytest.raises(Exception):
            attrs.evolve(inst, y=-1)

    def test_failure_leaves_old_instance_untouched(self):
        """
        If evolve fails, the old instance the caller holds is unchanged,
        field by field.
        """

        @attrs.define
        class C:
            x: int = attrs.field(validator=attrs.validators.gt(0))
            y: int = 2

        old = C(1, 2)
        with pytest.raises(Exception):
            attrs.evolve(old, x=-5)
        assert old.x == 1
        assert old.y == 2

    def test_unknown_name_rejected(self):
        """
        Evolving a name that doesn't exist raises TypeError and leaves
        the old instance unchanged.
        """

        @attrs.define
        class C:
            a: int
            b: int

        old = C(1, 2)
        with pytest.raises(TypeError):
            attrs.evolve(old, nonexistent=3)
        assert (old.a, old.b) == (1, 2)

    def test_chained_evolve_builds_on_previous(self):
        """
        Evolving twice in a row bases the second evolve on the first
        new instance, not on the original one.
        """

        @attrs.define
        class C:
            a: int
            b: int

        first = attrs.evolve(C(1, 2), a=10)
        second = attrs.evolve(first, b=20)

        assert (second.a, second.b) == (10, 20)

    def test_subclass_fields_are_kept(self):
        """
        Evolving a subclass instance keeps the fields the subclass adds
        on top of the parent.
        """

        @attrs.define
        class Parent:
            x: int

        @attrs.define
        class Child(Parent):
            y: int

        new = attrs.evolve(Child(1, 2), x=10)

        assert type(new) is Child
        assert (new.x, new.y) == (10, 2)

    def test_parent_rejects_subclass_only_field(self):
        """
        Evolving a parent instance can't write a field the parent
        doesn't have into it.
        """

        @attrs.define
        class Parent:
            x: int

        @attrs.define
        class Child(Parent):
            y: int

        old = Parent(1)
        with pytest.raises(TypeError):
            attrs.evolve(old, y=5)
        assert old.x == 1


class TestAsdict:
    def test_non_comparing_fields_included_by_default(self):
        """
        Fields that don't participate in comparison are still part of
        the unfolded mapping by default; only caller-filtered fields are
        dropped.
        """

        @attrs.define
        class C:
            a: int
            b: int = attrs.field(eq=False)

        inst = C(1, 2)

        assert attrs.asdict(inst) == {"a": 1, "b": 2}
        assert attrs.asdict(inst, filter=lambda a, v: a.name != "b") == {"a": 1}

    def test_dropped_fields_absent_and_order_matches_roster(self):
        """
        Filtered-out fields don't appear in the mapping; the remaining
        order matches the field roster.
        """

        @attrs.define
        class C:
            a: int
            b: int
            c: int

        d = attrs.asdict(C(1, 2, 3), filter=lambda a, v: a.name != "b")

        assert ["a", "c"] == list(d)

    def test_type_information_is_optional(self):
        """
        Unfolding can choose to retain collection types or not; either
        way the values themselves are still there.
        """

        @attrs.define
        class C:
            ts: tuple

        inst = C((1, 2))

        assert [1, 2] == attr.asdict(inst)["ts"]
        assert (1, 2) == attr.asdict(inst, retain_collection_types=True)["ts"]
        assert (1, 2) == attrs.asdict(inst)["ts"]

    def test_extra_attributes_ignored_by_default(self):
        """
        On dict (non-slots) classes, extra attributes attached to the
        instance are kept on it but aren't unfolded as fields.
        """

        @attrs.define(slots=False)
        class C:
            x: int

        inst = C(1)
        inst.extra = 99

        assert inst.extra == 99
        assert attrs.asdict(inst) == {"x": 1}
        assert attr.asdict(inst) == {"x": 1}

    def test_asdict_after_evolve_shows_new_values(self):
        """
        Unfolding an evolved instance shows the evolved values, not the
        old ones.
        """

        @attrs.define
        class C:
            a: int
            b: int

        new = attrs.evolve(C(1, 2), a=10)

        assert attrs.asdict(new) == {"a": 10, "b": 2}


class TestFields:
    def test_order_matches_definition(self):
        """
        The field roster is in definition order.
        """

        @attrs.define
        class C:
            a: int
            b: int
            c: int

        assert [f.name for f in attrs.fields(C)] == ["a", "b", "c"]

    def test_private_names_have_public_aliases(self):
        """
        Private fields are listed under their defined name while the
        generated __init__ uses the public alias; no temporary names
        leak to callers.
        """

        @attrs.define
        class C:
            _x: int

        (field,) = attrs.fields(C)

        assert field.name == "_x"
        assert field.alias == "x"
        inst = C(x=1)
        assert inst._x == 1
        assert attrs.evolve(inst, x=2)._x == 2


class TestSlots:
    def test_slots_reject_new_attribute_at_assignment(self):
        """
        With slots on, attaching a new name to an instance is rejected
        at assignment time.
        """

        @attrs.define
        class C:
            x: int

        inst = C(1)
        with pytest.raises(AttributeError):
            inst.new_name = 3

    def test_dict_class_keeps_extra_attribute(self):
        """
        With slots off, extra names can be attached to the instance.
        """

        @attrs.define(slots=False)
        class C:
            x: int

        inst = C(1)
        inst.extra = 2

        assert inst.extra == 2


class TestFactory:
    def test_fresh_container_per_instance(self):
        """
        Default factories produce a fresh value for every instance;
        instances never share a factory-made container.
        """

        @attrs.define
        class C:
            items: list = attrs.field(factory=list)

        first = C()
        second = C()

        assert first.items is not second.items
        first.items.append(1)
        assert second.items == []

    def test_failure_leaves_no_half_instance(self):
        """
        If a factory fails, the error propagates and no half-initialized
        instance is handed out.
        """

        def bad_factory():
            raise RuntimeError("boom")

        @attrs.define
        class C:
            x: list = attrs.field(factory=bad_factory)

        with pytest.raises(RuntimeError):
            C()


class TestConsistency:
    def test_eq_hash_ignore_asdict_keeps(self):
        """
        Comparison, hashing, and unfolding agree about a field that
        doesn't participate in comparison: comparison and hashing ignore
        it, unfolding still keeps it.
        """

        @attrs.frozen
        class C:
            a: int
            b: int = attrs.field(eq=False)

        first = C(1, 2)
        second = C(1, 3)

        assert first == second
        assert hash(first) == hash(second)
        assert attrs.asdict(first) == {"a": 1, "b": 2}

    def test_all_five_compose(self):
        """
        Evolve, unfold, roster, slots, and factories work together on
        one frozen class.
        """

        @attrs.frozen
        class C:
            name: str
            tags: list = attrs.field(factory=list)
            cached: int = attrs.field(default=0, eq=False)

        old = C("n")
        new = attrs.evolve(old, name="m")

        assert new is not old
        assert old.name == "n"
        assert [f.name for f in attrs.fields(C)] == ["name", "tags", "cached"]
        assert attrs.asdict(new) == {"name": "m", "tags": [], "cached": 0}
        with pytest.raises(AttributeError):
            new.extra = 1
