from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


class TestSearchEndpoint:
    def test_search_success(self, monkeypatch):
        mock_results = {
            "organic_results": [
                {"title": "Result 1", "snippet": "Snippet 1", "link": "http://1"},
                {"title": "Result 2", "snippet": "Snippet 2", "link": "http://2"},
                {"title": "Result 3", "snippet": "Snippet 3", "link": "http://3"},
            ]
        }

        mock_search = MagicMock()
        mock_search.get_dict.return_value = mock_results

        # Patch environment and GoogleSearch
        monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
        with patch("app.GoogleSearch", return_value=mock_search) as mock_google:
            response = client.post("/search", json={"query": "python"})

        assert response.status_code == 200
        expected = (
            "Title: Result 1\nSnippet: Snippet 1\nURL: http://1\n\n"
            "Title: Result 2\nSnippet: Snippet 2\nURL: http://2\n\n"
            "Title: Result 3\nSnippet: Snippet 3\nURL: http://3\n"
        )
        assert response.json()["results"] == expected.strip()
        mock_google.assert_called_once_with({"q": "python", "api_key": "test-key", "num": 3})
        mock_search.get_dict.assert_called_once()

    def test_search_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        response = client.post("/search", json={"query": "test"})

        assert response.status_code == 500
        assert response.json()["detail"] == "SERPAPI_API_KEY not found in environment variables"


class TestHealthCheck:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
