package com.d105.loanplease.store;

import com.d105.loanplease.ApiTest;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;

import static org.assertj.core.api.Assertions.*;

public class StoreApiTest extends ApiTest {

    @Test
    void 상점전체조회() {
        final var response = StoreSteps.상점전체조회();
        assertThat(response.statusCode()).isEqualTo(HttpStatus.OK.value());
    }
}
