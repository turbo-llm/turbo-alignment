from pydantic import BaseModel, field_validator


class DatasetRecord(BaseModel):
    id: str

    @field_validator('id', mode='before')
    def convert_int_id_to_str(cls, values: int) -> str:
        if isinstance(values, int):
            return str(values)
        return values
