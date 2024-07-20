from pydantic import BaseModel, Extra


class ExtraFieldsNotAllowedBaseModel(BaseModel):
    class Config:
        extra = Extra.forbid
        protected_namespaces = ()
