def assertDSL(test_obj, dsl, expected):
    dsl = "\n".join(sorted([line.strip() for line in dsl.split("\n") if line.strip()]))
    expected = "\n".join(
        sorted([line.strip() for line in expected.split("\n") if line.strip()])
    )
    print(dsl)
    test_obj.maxDiff = None
    test_obj.assertEqual(dsl, expected)
