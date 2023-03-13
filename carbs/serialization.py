from __future__ import (
    annotations,  # using this to get Postponed Evaluation of Annotations -- https://www.python.org/dev/peps/pep-0563/
)

import multiprocessing
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any
from typing import ContextManager
from typing import Dict
from typing import List
from typing import Set
from typing import Type
from typing import TypeVar
from typing import Union
from uuid import UUID

import attr

FREEZE_KEY = "$_frozen"

TC = TypeVar("TC", bound="Serializable")


_CLASS_KEY = "$type"

lock = multiprocessing.Lock()


@attr.s(hash=True, collect_by_mro=True)
class Serializable:
    # noinspection PyDataclass
    def __attrs_post_init__(self) -> None:
        self.__dict__[FREEZE_KEY] = True

    def __setattr__(self, item: str, value: Any):
        if self.__dict__.get(FREEZE_KEY):
            raise ValueError("instance is frozen; see: .mutable_clone()")
        else:
            self.__dict__[item] = value

    def mutable_clone(self: TC) -> ContextManager[TC]:
        freeze_list: List[Serializable] = []

        def thaw(attr_object: Serializable) -> Serializable:
            kwargs: Dict[str, Any] = {}
            for attribute in attr.fields(attr_object.__class__):
                k = attribute.name
                v = getattr(attr_object, k)
                if isinstance(v, Serializable):
                    kwargs[k] = thaw(v)
                elif isinstance(v, tuple) and len(v) > 0 and isinstance(v[0], Serializable):
                    kwargs[k] = tuple([thaw(x) for x in v])
                else:
                    kwargs[k] = deepcopy(v)

            # noinspection PyArgumentList
            attr_cloned = attr_object.__class__(**kwargs)
            attr_cloned.__dict__[FREEZE_KEY] = False
            freeze_list.append(attr_cloned)

            return attr_cloned

        @contextmanager
        def context():
            try:
                yield thaw(self)
            finally:
                for attr_cloned in freeze_list:
                    attr_cloned.__dict__[FREEZE_KEY] = True

        return context()

    def to_dict(self) -> dict:
        dump: Dict[str, Any] = {_CLASS_KEY: get_qualname_from_serializable_type(type(self))}
        for attribute in attr.fields(self.__class__):
            k = attribute.name
            v = getattr(self, k)
            dump[k] = _to_dict(v)
        return dump

    @classmethod
    def from_dict(cls: Type[TC], dump: dict, is_upgrade_allowed: bool = False) -> TC:
        """
        Using is_upgrade_allowed is dangerous:
        - Silently ignores keys that exist in the serialization but no longer in the object
          Could definitely lose data.
        - Silently ignores missing attributes, assuming they will have default values.
          This will likely blow up if there are no default values.
        The most likely case that this will be problematic is if you RENAMED an attribute
        There's no way for us to really detect that.
        I'd be ok with someone adding explicit "rename" support when they actually need it
        """
        kwargs = {}
        remaining_dict_keys = set(dump.keys())
        for attribute in attr.fields(cls):
            k = attribute.name
            if is_upgrade_allowed and k not in dump:
                # this enables us to load older ubjects when new default attributes have been added
                # if there is no default value, the later construction will fail, there is
                # no easy way around that without getting into more complex migration schemes
                continue
            if k not in remaining_dict_keys:
                raise Exception(f"Serialized object {cls} is missing key={k}")
            remaining_dict_keys.remove(k)
            kwargs[k] = _from_value(dump[k], is_upgrade_allowed)
        if _CLASS_KEY in remaining_dict_keys:
            remaining_dict_keys.remove(_CLASS_KEY)
        if len(remaining_dict_keys) > 0:
            # silently drops keys here!
            if not is_upgrade_allowed:
                bad_keys = sorted(remaining_dict_keys)
                raise ParamTypeError(f"Tried to load data for {cls} but got unexpected keys: {bad_keys}")
        # noinspection PyArgumentList
        return cls(**kwargs)


def _to_dict(v: Any) -> Any:
    # better to check for the existance of this attribute than to check isinstance
    # even works in jupyter notebooks when doing code reloading then..
    if hasattr(v, "to_dict"):
        result = v.to_dict()
    elif isinstance(v, tuple):
        if len(v) > 0:
            result = tuple([_to_dict(x) for x in v])
        else:
            result = v
    elif isinstance(v, Enum):
        result = {"value": v.value, _CLASS_KEY: get_qualname_from_serializable_type(type(v))}
    elif isinstance(v, UUID):
        result = dict(value=str(v))
        result[_CLASS_KEY] = UUID.__name__
    elif isinstance(v, datetime):
        result = dict(value=v.isoformat())
        result[_CLASS_KEY] = datetime.__name__
    else:
        assert isinstance(v, (float, int, bool, str, type(None))), "Unexpected type: " + str(v)
        result = v
    return result


def _from_value(v: Dict[str, Any], is_upgrade_allowed: bool) -> Any:
    result: Any
    if isinstance(v, Dict) and _CLASS_KEY in v:
        class_name = v[_CLASS_KEY]
        if class_name == UUID.__name__:
            result = UUID(v["value"])
        elif class_name == datetime.__name__:
            result = datetime.fromisoformat(v["value"])
        else:
            t = get_serializable_type_from_qualname(class_name)
            if issubclass(t, Enum):
                # noinspection PyArgumentList
                result = t(value=v["value"])  # type: ignore
            else:
                result = t.from_dict(v, is_upgrade_allowed=is_upgrade_allowed)
    # TAKS c00f9a38-bd11-4e14-97d8-02b9bf843ca6: delete this after we get rid of OldSerializable
    elif isinstance(v, Dict) and "_serializable_type" in v:
        t = get_serializable_type_from_qualname(v["_serializable_type"])
        result = t.from_dict(v, is_upgrade_allowed=is_upgrade_allowed)
    elif isinstance(v, (list, tuple)):
        if len(v) == 0:
            result = tuple()
        else:
            inner_objects = []
            for inner_value in v:
                inner_objects.append(_from_value(inner_value, is_upgrade_allowed))
            result = tuple(inner_objects)
    else:
        assert isinstance(v, (float, int, bool, str, type(None))), "Unexpected type: " + str(v)
        result = v
    return result


qualname_to_serializable_type: Dict[str, Type[Serializable]] = {}
serializable_type_to_qualname: Dict[Type[Serializable], str] = {}
# this is here because some libraries make enums with the same names, but we dont use those, so, fine
unserializable_types: Set[str] = set()


class ParamTypeError(TypeError):
    pass


def _get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(_get_all_subclasses(subclass))
    return all_subclasses


def _compute_unserializable(qualnames: List[str]) -> None:
    global unserializable_types

    if not unserializable_types:
        already_seen_names = set()
        for qualname in qualnames:
            if qualname in already_seen_names:
                unserializable_types.add(qualname)
            already_seen_names.add(qualname)


def _dedupe_subclasses_by_qualname(subclasses: List[type]) -> List[type]:
    qualname_to_serializable_type = {x.__qualname__: x for x in subclasses}
    return list(qualname_to_serializable_type.values())


def _dedupe_subclasses_by_id(subclasses: List[type]) -> List[type]:
    id_to_serializable_type = {id(x): x for x in subclasses}
    return list(id_to_serializable_type.values())


def get_all_serializable_classes() -> List[type]:
    # we don't reload the Enum class (since it's standard library) so we end up having duplicate subclasses for Enums
    enum_subclasses = _dedupe_subclasses_by_qualname(_get_all_subclasses(Enum))
    # we want to make sure we have classes with different names which is why we need to dedupe with the ids
    serializable_subclasses = _dedupe_subclasses_by_id(_get_all_subclasses(Serializable))
    return serializable_subclasses + enum_subclasses


def get_serializable_type_from_qualname(qualname: str) -> Type[Serializable]:
    global qualname_to_serializable_type
    global unserializable_types
    if not qualname_to_serializable_type:
        # makes our globals thread safe and hopefully free from weird race conditions
        with lock:
            if not qualname_to_serializable_type:
                all_serializable_classes = get_all_serializable_classes()
                qualname_to_serializable_type = {x.__qualname__: x for x in all_serializable_classes}
                _compute_unserializable([x.__qualname__ for x in all_serializable_classes])
    if qualname in unserializable_types:
        raise Exception(f"{qualname} cannot be deserialized because there are multiple definitions")
    return qualname_to_serializable_type[qualname]


def get_qualname_from_serializable_type(serializable_type: type) -> str:
    global serializable_type_to_qualname
    global unserializable_types
    if not serializable_type_to_qualname:
        # makes our globals thread safe and hopefully free from weird race conditions
        with lock:
            if not serializable_type_to_qualname:
                all_serializable_classes = get_all_serializable_classes()
                serializable_type_to_qualname = {x: x.__qualname__ for x in all_serializable_classes}
                _compute_unserializable(list(serializable_type_to_qualname.values()))
    result = serializable_type_to_qualname.get(serializable_type, serializable_type.__qualname__)
    if result in unserializable_types:
        raise Exception(f"{result} cannot be serialized because there are multiple definitions")
    return result


T = TypeVar("T", bool, int, float, str, tuple)
DictNest = Dict[str, Union[T, Dict[str, Any]]]
DictFlat = Dict[str, T]


def flatten_dict(d: DictNest, prefix: str = "") -> DictFlat:
    flattened_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flattened_dict.update(flatten_dict(v, f"{prefix}{k}."))
        else:
            flattened_dict[f"{prefix}{k}"] = v
    return flattened_dict


def inflate_dict(d: DictFlat) -> DictNest:
    inflated_dict: DictNest = {}
    for key, value in d.items():
        parts = key.split(".")
        d = inflated_dict
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return inflated_dict
