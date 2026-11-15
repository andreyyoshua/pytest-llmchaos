from llmchaos import errors

ALL_ERRORS = [
    errors.CassetteMissingError,
    errors.CassetteMismatchError,
    errors.UnexpectedCallError,
    errors.UnusedFaultError,
    errors.FaultNotApplicableError,
    errors.SecretLeakError,
]


def test_every_error_shares_the_base_class():
    for error in ALL_ERRORS:
        assert issubclass(error, errors.LLMChaosError)


def test_base_class_is_not_an_http_library_error():
    # SDKs that only retry network errors must let ours propagate unchanged.
    assert errors.LLMChaosError.__mro__ == (errors.LLMChaosError, Exception, BaseException, object)
