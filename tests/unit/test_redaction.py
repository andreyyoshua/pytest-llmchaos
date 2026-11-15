from llmchaos import redaction


def test_filter_headers_keeps_only_the_allowlist_and_lowercases():
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-x",
        "X-Api-Key": "k",
        "anthropic-version": "2023-06-01",
    }
    assert redaction.filter_headers(headers, redaction.REQUEST_HEADER_ALLOWLIST) == {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }


def test_find_secrets_reports_key_patterns_with_their_path():
    document = {
        "interactions": [
            {
                "request": {
                    "body": {"messages": [{"content": "use sk-proj-abcdefghijklmnopqrstuvwxyz"}]}
                }
            }
        ]
    }
    assert redaction.find_secrets(document, environ={}) == [
        "$.interactions[0].request.body.messages[0].content"
    ]


def test_find_secrets_ignores_words_that_merely_contain_sk_dash():
    document = {"text": "risk-assessment-for-the-quarterly-report"}
    assert redaction.find_secrets(document, environ={}) == []


def test_find_secrets_reports_environment_key_values():
    environ = {"ANTHROPIC_API_KEY": "plain-secret-value"}
    document = {"a": ["x", "has plain-secret-value inside"]}
    assert redaction.find_secrets(document, environ=environ) == ["$.a[1]"]


def test_find_secrets_ignores_short_environment_values():
    assert redaction.find_secrets({"a": "abc"}, environ={"OPENAI_API_KEY": "abc"}) == []
