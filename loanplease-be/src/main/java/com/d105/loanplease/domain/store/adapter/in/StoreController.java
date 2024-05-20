package com.d105.loanplease.domain.store.adapter.in;

import com.d105.loanplease.domain.store.application.port.in.ItemUseCase;
import com.d105.loanplease.domain.store.application.port.in.LoanUseCase;
import com.d105.loanplease.domain.store.application.service.request.ChooseLoanRequest;
import com.d105.loanplease.domain.store.application.service.request.PurchaseItemRequest;
import com.d105.loanplease.domain.store.application.service.request.PurchaseLoanRequest;
import com.d105.loanplease.domain.store.application.service.response.*;
import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.store.domain.Loan;
import io.swagger.v3.oas.annotations.Operation;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("api/store")
@Slf4j
public class StoreController {

    @Autowired
    private LoanUseCase loanUseCase;

    @Autowired
    private ItemUseCase itemUseCase;

    @Operation(summary = "게임 대출 상품 선택", description = "게임에서 사용할 대출 상품을 선택합니다.")
    @PutMapping("/choose-loan")
    public ResponseEntity<ChooseLoanResponse> chooseLoan(@RequestBody ChooseLoanRequest request) {

        log.info(String.valueOf(request.getSlot_1()));
        log.info(String.valueOf(request.getSlot_2()));
        log.info(String.valueOf(request.getSlot_3()));
        log.info(String.valueOf(request.getSlot_4()));
        log.info(String.valueOf(request.getSlot_5()));

        return loanUseCase.changeSlot(request.getSlot_1(),
                request.getSlot_2(),
                request.getSlot_3(),
                request.getSlot_4(),
                request.getSlot_5());
    }

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
    public ResponseEntity<PurchaseSlotResponse> purchaseSlot() {
        return itemUseCase.expandSlot();
    }

    @Operation(summary = "대출 상품 구매", description = "대출 상품을 구매합니다.")
    @PostMapping("/items/loan")
    public ResponseEntity<PurchaseLoanResponse> purchaseLoan(@RequestBody PurchaseLoanRequest request) {
        return loanUseCase.purchaseLoan(request.getLoanId());
    }

    @Operation(summary = "일회성 아이템 구매", description = "게임 시간 추가, VIP, 1회 방어권 아이템을 구매합니다.")
    @PostMapping("/items/oneoff")
    public ResponseEntity<PurchaseItemResponse> purchaseItem(@RequestBody PurchaseItemRequest request) {
        log.info(String.valueOf(request.getItemId())+" "+String.valueOf(request.getItemCount()));
        return itemUseCase.purchaseItem(request.getItemId(), request.getItemCount());
    }
}
