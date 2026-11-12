import importlib.util
import base64
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "blackwell/ocr_adapter.py"


class OvisOcrTests(unittest.TestCase):
    def test_adapter_preserves_private_ocr_contract(self):
        spec = importlib.util.spec_from_file_location("ocr_adapter", MODULE)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        image = base64.b64encode(b"\x89PNG\r\n\x1a\nplaceholder").decode()
        payload = adapter.build_request(image, "Read the text.", "ATH-MaaS/OvisOCR3")
        self.assertEqual(payload["model"], "ATH-MaaS/OvisOCR3")
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "image_url")
        self.assertTrue(payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/"))
        self.assertEqual(payload["messages"][0]["content"][1]["text"], "Read the text.")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        result = adapter.to_ocr_response({"choices": [{"message": {"content": "Invoice 42"}}]},
                                         "ATH-MaaS/OvisOCR3")
        self.assertEqual(result, {"model": "ATH-MaaS/OvisOCR3", "blocks": [
            {"text": "Invoice 42", "bbox": None, "confidence": None}]})



if __name__ == "__main__":
    unittest.main()
