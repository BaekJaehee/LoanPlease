package com.d105.loanplease.domain.store.application.service;

import com.d105.loanplease.domain.store.application.port.in.ItemUseCase;
import com.d105.loanplease.domain.store.application.port.out.ItemPort;
import com.d105.loanplease.domain.store.application.service.response.PurchaseItemResponse;
import com.d105.loanplease.domain.store.application.service.response.PurchaseSlotResponse;
import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.entity.UserItem;
import com.d105.loanplease.domain.user.repository.UserItemRepository;
import com.d105.loanplease.domain.user.repository.UserRepository;
import com.d105.loanplease.global.util.Constant;
import com.d105.loanplease.global.util.SecurityUtil;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Slf4j
@Service
@AllArgsConstructor
@Transactional(readOnly = true)
public class ItemService implements ItemUseCase {
    private final SecurityUtil securityUtil;
    @Autowired
    private ItemPort itemPort;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private UserItemRepository userItemRepository;

    @Override
    public List<Item> inquiryAllItems() {
        return itemPort.findAll();
    }

    @Override
    public Item inquiryItemById(final Long itemId) {
        return itemPort.findById(itemId);
    }

    @Override
    @Transactional
    public ResponseEntity<PurchaseSlotResponse> expandSlot() {
        User user = securityUtil.getCurrentUserDetails();
        /**
         * 유저의 슬롯 구매
         * 1. 유저의 슬롯 개수 확인
         * 2. 유저의 보유 포인트 확인
         * 3. 유저의 포인트 차감 + 슬롯 확장
         */
        user.expandSlot();
        PurchaseSlotResponse response = new PurchaseSlotResponse(user.getPoint(), user.getSlotNum());

        return ResponseEntity.ok(response);
    }

    @Override
    @Transactional
    public ResponseEntity<PurchaseItemResponse> purchaseItem(final Long itemId, final Integer itemCount) {
        Item item = itemPort.findById(itemId);
        User user = securityUtil.getCurrentUserDetails();

        Long userItemId = user.hasItemHistory(itemId);
        log.info("userItemID: "+String.valueOf(userItemId));

        UserItem userItem = userItemRepository.findById(userItemId)
                .orElseThrow(() -> new IllegalArgumentException("해당 보유 아이템은 없는 아이템입니다."));
        userItem.purchaseItem(item.getPrice(), itemCount, user);

        List<UserItem> userItemList = user.getUserItemList();
        PurchaseItemResponse response = new PurchaseItemResponse(user.getPoint(), userItemList);

        return ResponseEntity.ok(response);
    }
}
