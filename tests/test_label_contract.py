import json
import pytest
from conversation_target_tagger.tagging import parse_and_validate


@pytest.mark.parametrize("index,label", [(True,"self"),(1.0,"self"),(1,"project:"),(1,"person_external:   "),(1,"self:someone"),(1,"ambiguous:maybe")])
def test_malformed_labels_and_indices_are_rejected(index,label):
    with pytest.raises(ValueError):
        parse_and_validate(json.dumps([{"gidx":index,"targets":[label]}]),(1,))
