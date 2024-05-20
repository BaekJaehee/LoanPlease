package com.d105.loanplease.domain.store.application.port.in;

import com.d105.loanplease.domain.store.application.service.response.PurchaseItemResponse;
import com.d105.loanplease.domain.store.application.service.response.PurchaseSlotResponse;
import com.d105.loanplease.domain.store.domain.Item;
import org.springframework.http.ResponseEntity;

import java.util.List;

public interface ItemUseCase {

    List<Item> inquiryAllItems();
    Item inquiryItemById(Long itemId);
    ResponseEntity<PurchaseSlotResponse> expandSlot();
    ResponseEntity<PurchaseItemResponse> purchaseItem(Long itemId, Integer itemCount);
}
