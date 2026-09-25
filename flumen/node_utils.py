from __future__ import annotations

from typing import Iterable, Optional


def resolve_socket(sockets, *, name=None, index=None, identifier=None):
    """Select an available socket without guessing between duplicate UI names."""
    if sum(x is not None for x in (name, index, identifier)) != 1:
        raise ValueError('Provide exactly one socket selector')
    available = lambda s: s.enabled and not s.is_unavailable
    if index is not None:
        if not 0 <= index < len(sockets) or not available(sockets[index]):
            raise ValueError(f'Unavailable socket index: {index}')
        return sockets[index]
    matches = [s for s in sockets if available(s) and
               (s.identifier == identifier if identifier is not None else s.name == name)]
    if len(matches) != 1:
        raise ValueError(f'Expected one socket for {identifier or name!r}; found {len(matches)}')
    return matches[0]


def clear_nodes(tree):
    for node in list(tree.nodes):
        tree.nodes.remove(node)


def socket_by_name(sockets, names: Iterable[str], fallback: Optional[int] = None):
    """Find a socket defensively by one of several possible UI names."""
    for name in names:
        matches = [s for s in sockets if s.name == name and s.enabled and not s.is_unavailable]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous socket {name!r}; use an explicit index')
    if fallback is not None and 0 <= fallback < len(sockets):
        return sockets[fallback]
    available = ", ".join(s.name for s in sockets)
    raise KeyError(f"Socket not found. Tried {tuple(names)}. Available: {available}")


def new_node(tree, bl_idname: str, label: str, x: float, y: float):
    node = tree.nodes.new(bl_idname)
    node.label = label
    node.name = label
    node.location = (x, y)
    return node


def link(tree, from_node, from_names, to_node, to_names, from_fallback=None, to_fallback=None,
         from_index=None, to_index=None):
    out_socket = (resolve_socket(from_node.outputs, index=from_index) if from_index is not None
                  else socket_by_name(from_node.outputs, from_names, from_fallback))
    in_socket = (resolve_socket(to_node.inputs, index=to_index) if to_index is not None
                 else socket_by_name(to_node.inputs, to_names, to_fallback))
    tree.links.new(out_socket, in_socket)
    return in_socket


def set_input(node, names, value, fallback=None, index=None):
    sock = (resolve_socket(node.inputs, index=index) if index is not None
            else socket_by_name(node.inputs, names, fallback))
    if hasattr(sock, "default_value"):
        sock.default_value = value
    return sock


def add_interface_socket(tree, name: str, in_out: str, socket_type: str, *, default=None,
                         min_value=None, max_value=None, description: str = ""):
    sock = tree.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)
    if description and hasattr(sock, "description"):
        sock.description = description
    if default is not None and hasattr(sock, "default_value"):
        sock.default_value = default
    if min_value is not None and hasattr(sock, "min_value"):
        sock.min_value = min_value
    if max_value is not None and hasattr(sock, "max_value"):
        sock.max_value = max_value
    return sock
