# Pascal port: SM70 policy-op registration is a no-op (no SM70 native ABI).
# The registered ops are only invoked from disabled SM70 fast-paths.
def register_policy_op(name, schema, operation, fake):
    return operation


__all__ = ["register_policy_op"]
