package com.d105.loanplease.domain.store.adapter.in;

import com.d105.loanplease.domain.store.application.port.in.ItemUseCase;
import com.d105.loanplease.domain.store.application.port.in.LoanUseCase;
import com.d105.loanplease.domain.store.application.service.response.InquiryStoreResponse;
import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.store.domain.Loan;
import io.swagger.v3.oas.annotations.Operation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/store")
public class StoreController {

    @Autowired
    private LoanUseCase loanUseCase;

    @Autowired
    private ItemUseCase itemUseCase;


    @Operation(summary = "게임 대출 상품 선택", description = "게임에서 사용할 대출 상품을 선택합니다.")
    @PostMapping("/choose-loan")
    public void chooseLoan() {}

    @Operation(summary = "상점 목록 조회", description = "상점의 슬롯 확장, 일회성 아이템, 대출 상품 목록을 조회합니다.")
    @GetMapping("/items")
    public ResponseEntity<InquiryStoreResponse> inquiryStore() {
        List<Loan> loans = loanUseCase.inquiryAllLoans();
        List<Item> items = itemUseCase.inquiryAllItems();

        InquiryStoreResponse response = new InquiryStoreResponse();

        for(Loan loan: loans) {
            response.addLoan(loan);
        }

        for(Item item: items) {
            response.addItem(item);
        }

        return ResponseEntity.ok(response);
    }

    @Operation(summary = "슬롯 구매", description = "슬롯 확장 아이템을 구매합니다.")
    @PostMapping("/items/slot")
    public void purchaseSlot() {
//        itemUseCase.expandSlot();
    }

    @Operation(summary = "대출 상품 구매", description = "대출 상품을 구매합니다.")
    @PostMapping("/items/loan")
    public void purchaseLoan() {
//        loanUseCase.purchaseLoan();
    }

    @Operation(summary = "일회성 아이템 구매", description = "게임 시간 추가, VIP, 1회 방어권 아이템을 구매합니다.")
    @PostMapping("/items/oneoff")
    public void purchaseItem() {}
}