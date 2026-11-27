"""Step 17: complete-system smoke tests for the Flask prototype."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app, users


class HospitalReadmissionAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False

    def setUp(self):
        users.clear()
        self.client = app.test_client()

    def tearDown(self):
        users.clear()

    def register(self):
        return self.client.post(
            "/register",
            data={
                "name": "System Tester",
                "email": "system@example.com",
                "password": "secret1",
            },
            follow_redirects=False,
        )

    def test_home_requires_login(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/login")

    def test_register_and_login_flow(self):
        response = self.register()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/")

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Hospital Readmission Risk Prediction", response.data)

        self.client.get("/logout")
        response = self.client.post(
            "/login",
            data={"email": "system@example.com", "password": "secret1"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/")

    def test_assessment_accepts_optional_fields(self):
        self.register()
        response = self.client.post(
            "/",
            data={
                "age": "67",
                "gender": "M",
                "bmi": "26.4",
                "sepsis_label": "1",
                "sofa_score": "9",
                "apache_iv": "45",
                "hospital_admit_source": "Transfer",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"estimated 30-day", response.data)
        self.assertIn(b"Follow-up considerations", response.data)

    def test_registration_page_removed_promotional_copy(self):
        response = self.client.get("/register")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Make room for", response.data)
        self.assertNotIn(b"Create a local prototype account", response.data)

    def test_logout_protects_home_again(self):
        self.register()
        response = self.client.get("/logout", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/login")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/login")


if __name__ == "__main__":
    unittest.main(verbosity=2)
