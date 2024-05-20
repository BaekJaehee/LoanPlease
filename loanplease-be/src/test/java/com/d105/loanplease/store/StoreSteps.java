package com.d105.loanplease.store;

import com.d105.loanplease.domain.store.application.service.response.InquiryStoreResponse;
import io.restassured.RestAssured;
import io.restassured.response.ExtractableResponse;
import io.restassured.response.Response;
import org.springframework.http.MediaType;

public class StoreSteps {

    public static ExtractableResponse<Response> 상점전체조회() {
        return RestAssured.given().log().all()
                .when()
                .get("/store/items")
                .then()
                .log().all().extract();
    }
}
